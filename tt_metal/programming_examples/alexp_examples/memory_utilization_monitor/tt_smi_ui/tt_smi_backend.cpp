// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "tt_smi_backend.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <chrono>

// TT-UMD includes
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/topology/topology_discovery.hpp"
#include "umd/device/chip/chip.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/firmware/firmware_info_provider.hpp"

using namespace tt::umd;

namespace tt_smi {

// Shared memory structure (must match memory_stats_shm.hpp)
struct SHMDeviceMemoryRegion {
    uint32_t version;
    uint32_t num_active_processes;
    uint64_t last_update_timestamp;
    std::atomic<uint32_t> reference_count;

    uint64_t board_serial;
    uint64_t asic_id;
    int32_t device_id;

    std::atomic<uint64_t> total_dram_allocated;
    std::atomic<uint64_t> total_l1_allocated;
    std::atomic<uint64_t> total_l1_small_allocated;
    std::atomic<uint64_t> total_trace_allocated;
    std::atomic<uint64_t> total_cb_allocated;

    static constexpr size_t MAX_CHIPS_PER_DEVICE = 16;
    struct ChipStats {
        uint32_t chip_id;
        uint32_t is_remote;
        std::atomic<uint64_t> dram_allocated;
        std::atomic<uint64_t> l1_allocated;
        std::atomic<uint64_t> l1_small_allocated;
        std::atomic<uint64_t> trace_allocated;
        std::atomic<uint64_t> cb_allocated;
    } chip_stats[MAX_CHIPS_PER_DEVICE];

    static constexpr size_t MAX_PROCESSES = 64;
    struct ProcessStats {
        pid_t pid;
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
        uint64_t last_update_timestamp;
        char process_name[64];
    } processes[MAX_PROCESSES];

    uint8_t padding[64];
};

// Topology cache
struct TopologyCache {
    std::unique_ptr<ClusterDescriptor> cluster_descriptor;
    std::map<uint64_t, std::unique_ptr<TTDevice>>
        devices;  // Changed from Chip to TTDevice to match TopologyDiscovery API
    bool initialized = false;
    int consecutive_errors = 0;
    static constexpr int ERROR_THRESHOLD = 3;  // Re-init after 3 consecutive errors

    void initialize() {
        if (initialized) {
            return;
        }
        try {
            TopologyDiscoveryOptions options;
            options.no_remote_discovery = false;
            auto [descriptor, chip_map] = TopologyDiscovery::discover(options);
            cluster_descriptor = std::move(descriptor);
            devices = std::move(chip_map);
            initialized = true;
            consecutive_errors = 0;
        } catch (...) {
            initialized = false;
        }
    }

    void reinitialize() {
        // Clear existing state
        devices.clear();
        cluster_descriptor.reset();
        initialized = false;
        consecutive_errors = 0;
        // Re-discover
        initialize();
    }

    void record_error() { consecutive_errors++; }

    void record_success() { consecutive_errors = 0; }

    bool should_reinitialize() const { return consecutive_errors >= ERROR_THRESHOLD; }
};

static TopologyCache g_topology_cache;

// Format bytes with units
std::string format_bytes(uint64_t bytes) {
    if (bytes == 0) {
        return "0B";
    }
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int unit_idx = 0;
    double value = bytes;
    while (value >= 1024.0 && unit_idx < 4) {
        value /= 1024.0;
        unit_idx++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << value << units[unit_idx];
    return oss.str();
}

// Get architecture name
static std::string get_arch_name(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "Grayskull";
        case tt::ARCH::WORMHOLE_B0: return "Wormhole_B0";
        case tt::ARCH::BLACKHOLE: return "Blackhole";
        default: return "Unknown";
    }
}

// Check if process is alive
static bool is_process_alive(pid_t pid) {
    if (pid <= 0) {
        return false;
    }
    return (kill(pid, 0) == 0);
}

// Read memory from SHM
static bool read_memory_from_shm(Device& dev) {
    std::string shm_name = "/tt_device_" + std::to_string(dev.asic_id) + "_memory";
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd < 0) {
        return false;
    }

    auto* region =
        static_cast<SHMDeviceMemoryRegion*>(mmap(nullptr, sizeof(SHMDeviceMemoryRegion), PROT_READ, MAP_SHARED, fd, 0));

    if (region == MAP_FAILED) {
        close(fd);
        return false;
    }

    // Read aggregate stats
    dev.used_dram = region->total_dram_allocated.load(std::memory_order_relaxed);
    dev.used_l1 = region->total_l1_allocated.load(std::memory_order_relaxed);
    dev.used_l1_small = region->total_l1_small_allocated.load(std::memory_order_relaxed);
    dev.used_trace = region->total_trace_allocated.load(std::memory_order_relaxed);
    dev.used_cb = region->total_cb_allocated.load(std::memory_order_relaxed);

    // Read per-process data
    // IMPORTANT: Iterate through ALL entries, not just up to num_active_processes
    // The processes array can have gaps (entries with pid=0), so we must check all entries
    dev.processes.clear();
    for (uint32_t i = 0; i < SHMDeviceMemoryRegion::MAX_PROCESSES; i++) {
        auto& proc_entry = region->processes[i];
        if (proc_entry.pid > 0 && is_process_alive(proc_entry.pid)) {
            ProcessMemory proc;
            proc.pid = proc_entry.pid;
            proc.name = std::string(proc_entry.process_name);
            proc.dram_allocated = proc_entry.dram_allocated;
            proc.l1_allocated = proc_entry.l1_allocated;
            proc.l1_small_allocated = proc_entry.l1_small_allocated;
            proc.trace_allocated = proc_entry.trace_allocated;
            proc.cb_allocated = proc_entry.cb_allocated;
            dev.processes.push_back(proc);
        }
    }

    munmap(region, sizeof(SHMDeviceMemoryRegion));
    close(fd);
    dev.has_shm = true;
    return true;
}

// Set memory sizes from SoC descriptor
static void set_memory_sizes(Device& dev, TTDevice* tt_device) {
    if (!tt_device) {
        dev.total_dram = 0;
        dev.total_l1 = 0;
        return;
    }

    try {
        // Create SocDescriptor from architecture to get memory sizes
        tt::ARCH arch = tt_device->get_arch();

        // Use the chip info from tt_device if available
        tt::ChipInfo chip_info = tt_device->get_chip_info();
        SocDescriptor soc_desc(arch, chip_info);

        // Get DRAM info: num_channels * bank_size
        size_t num_dram_channels = soc_desc.get_num_dram_channels();
        uint64_t dram_bank_size = soc_desc.dram_bank_size;
        dev.total_dram = num_dram_channels * dram_bank_size;

        // Get L1 info: l1_size_per_core * num_tensix_cores
        uint32_t l1_size_per_core = soc_desc.worker_l1_size;
        auto tensix_grid = soc_desc.get_grid_size(tt::CoreType::TENSIX);
        dev.total_l1 = (uint64_t)l1_size_per_core * tensix_grid.x * tensix_grid.y;
    } catch (const std::exception& e) {
        // Log error but fallback to zero
        std::cerr << "Error getting memory sizes for device: " << e.what() << std::endl;
        dev.total_dram = 0;
        dev.total_l1 = 0;
    } catch (...) {
        // Fallback to zero on error
        std::cerr << "Unknown error getting memory sizes for device" << std::endl;
        dev.total_dram = 0;
        dev.total_l1 = 0;
    }
}

// Query telemetry
static bool query_telemetry(Device& dev) {
    auto device_it = g_topology_cache.devices.find(dev.chip_id);
    if (device_it == g_topology_cache.devices.end()) {
        dev.telemetry.status = "No device";
        dev.telemetry.available = false;
        g_topology_cache.record_error();
        return false;
    }

    TTDevice* tt_device = device_it->second.get();
    if (!tt_device) {
        dev.telemetry.status = "No device";
        dev.telemetry.available = false;
        g_topology_cache.record_error();
        return false;
    }

    try {
        auto firmware_info = tt_device->get_firmware_info_provider();
        if (!firmware_info) {
            dev.telemetry.status = "No FW info";
            dev.telemetry.available = false;
            g_topology_cache.record_error();
            return false;
        }

        double temp = firmware_info->get_asic_temperature();
        if (temp >= -50.0 && temp <= 100.0) {
            dev.telemetry.temperature = temp;
        }

        auto tdp = firmware_info->get_tdp();
        if (tdp.has_value() && tdp.value() > 0 && tdp.value() <= 500) {
            dev.telemetry.power = tdp.value();
        }

        auto vcore = firmware_info->get_vcore();
        if (vcore.has_value() && vcore.value() > 0 && vcore.value() <= 10000) {
            dev.telemetry.voltage_mv = vcore.value();
        }

        auto tdc = firmware_info->get_tdc();
        if (tdc.has_value() && tdc.value() > 0 && tdc.value() <= 500) {
            dev.telemetry.current_ma = tdc.value() * 1000;  // Convert amps to milliamps
        }

        auto aiclk = firmware_info->get_aiclk();
        if (aiclk.has_value() && aiclk.value() > 0 && aiclk.value() <= 1500) {
            dev.telemetry.aiclk_mhz = aiclk.value();
        }

        dev.telemetry.status = "OK";
        dev.telemetry.available = true;
        g_topology_cache.record_success();
        return true;

    } catch (const std::exception& e) {
        dev.telemetry.status = "Error";
        dev.telemetry.available = false;
        g_topology_cache.record_error();
        return false;
    }
}

// Public API implementations
std::vector<Device> enumerate_devices() {
    std::vector<Device> devices;

    // Check if we need to re-initialize due to errors (e.g., after device reset)
    if (g_topology_cache.should_reinitialize()) {
        g_topology_cache.reinitialize();
    }

    if (!g_topology_cache.initialized) {
        g_topology_cache.initialize();
    }

    if (g_topology_cache.initialized && !g_topology_cache.devices.empty()) {
        for (const auto& [chip_id, tt_device_ptr] : g_topology_cache.devices) {
            Device dev;
            dev.chip_id = chip_id;
            dev.arch_name = "Unknown";

            TTDevice* tt_device = tt_device_ptr.get();
            if (!tt_device) {
                continue;
            }

            try {
                dev.is_remote = tt_device->is_remote();
            } catch (...) {
                dev.is_remote = false;
            }

            try {
                dev.arch_name = get_arch_name(tt_device->get_arch());
                set_memory_sizes(dev, tt_device);

                uint64_t board_serial = tt_device->get_board_id();
                const auto& chip_info = tt_device->get_chip_info();
                dev.asic_location = chip_info.asic_location;

                // Compute display ID and asic_id
                if (chip_info.board_type == tt::BoardType::UBB && tt_device->get_pci_device()) {
                    uint16_t pci_bus = tt_device->get_pci_device()->get_device_info().pci_bus;
                    static const std::vector<uint16_t> tray_bus_ids = {0xC0, 0x80, 0x00, 0x40};
                    uint16_t bus_upper = pci_bus & 0xF0;
                    auto tray_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_upper);

                    if (tray_it != tray_bus_ids.end()) {
                        dev.tray_id = static_cast<uint32_t>(tray_it - tray_bus_ids.begin()) + 1;
                        dev.chip_in_tray = pci_bus & 0x0F;
                        uint32_t asic_location_composite = (dev.tray_id << 4) | dev.chip_in_tray;
                        dev.asic_id = (board_serial << 8) | asic_location_composite;

                        char buf[16];
                        snprintf(buf, sizeof(buf), "T%u:N%u", dev.tray_id, dev.chip_in_tray);
                        dev.display_id = buf;
                    } else {
                        dev.asic_id = (board_serial << 8) | dev.asic_location;
                        char buf[16];
                        snprintf(buf, sizeof(buf), "%llx", (unsigned long long)chip_id);
                        dev.display_id = buf;
                    }
                } else {
                    dev.asic_id = (board_serial << 8) | dev.asic_location;
                    char buf[16];
                    if (dev.is_remote) {
                        snprintf(buf, sizeof(buf), "%llxR", (unsigned long long)chip_id);
                    } else {
                        snprintf(buf, sizeof(buf), "%llx", (unsigned long long)chip_id);
                    }
                    dev.display_id = buf;
                }
            } catch (...) {
            }

            // Read memory stats
            read_memory_from_shm(dev);

            devices.push_back(dev);
        }
    }

    return devices;
}

bool update_device_telemetry(Device& device) { return query_telemetry(device); }

bool update_device_memory(Device& device) { return read_memory_from_shm(device); }

int cleanup_dead_processes() {
    int total_cleaned = 0;
    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.find("tt_device_") == 0 && filename.find("_memory") != std::string::npos) {
            std::string shm_name = "/" + filename;
            int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
            if (fd < 0) {
                continue;
            }

            auto* region = static_cast<SHMDeviceMemoryRegion*>(
                mmap(nullptr, sizeof(SHMDeviceMemoryRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

            if (region != MAP_FAILED) {
                for (uint32_t i = 0; i < SHMDeviceMemoryRegion::MAX_PROCESSES; i++) {
                    pid_t pid = region->processes[i].pid;
                    if (pid > 0 && !is_process_alive(pid)) {
                        auto& proc = region->processes[i];
                        region->total_dram_allocated.fetch_sub(proc.dram_allocated, std::memory_order_relaxed);
                        region->total_l1_allocated.fetch_sub(proc.l1_allocated, std::memory_order_relaxed);
                        region->total_l1_small_allocated.fetch_sub(proc.l1_small_allocated, std::memory_order_relaxed);
                        region->total_trace_allocated.fetch_sub(proc.trace_allocated, std::memory_order_relaxed);
                        region->total_cb_allocated.fetch_sub(proc.cb_allocated, std::memory_order_relaxed);
                        memset(&region->processes[i], 0, sizeof(SHMDeviceMemoryRegion::ProcessStats));
                        total_cleaned++;
                    }
                }
                munmap(region, sizeof(SHMDeviceMemoryRegion));
            }
            close(fd);
        }
    }
    closedir(dir);
    return total_cleaned;
}

}  // namespace tt_smi
