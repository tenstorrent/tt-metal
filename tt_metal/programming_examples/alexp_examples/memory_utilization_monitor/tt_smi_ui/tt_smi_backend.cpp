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
    std::atomic<uint64_t> total_kernel_allocated;

    static constexpr size_t MAX_CHIPS_PER_DEVICE = 16;
    struct ChipStats {
        uint32_t chip_id;
        uint32_t is_remote;
        std::atomic<uint64_t> dram_allocated;
        std::atomic<uint64_t> l1_allocated;
        std::atomic<uint64_t> l1_small_allocated;
        std::atomic<uint64_t> trace_allocated;
        std::atomic<uint64_t> cb_allocated;
        std::atomic<uint64_t> kernel_allocated;
    } chip_stats[MAX_CHIPS_PER_DEVICE];

    static constexpr size_t MAX_PROCESSES = 64;
    struct ProcessStats {
        pid_t pid;
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
        uint64_t kernel_allocated;
        uint64_t last_update_timestamp;
        char process_name[64];
    } processes[MAX_PROCESSES];

    uint8_t padding[64];
};

// Topology cache
struct TopologyCache {
    std::unique_ptr<ClusterDescriptor> cluster_descriptor;
    std::map<uint64_t, std::unique_ptr<Chip>> chips;
    bool initialized = false;

    void initialize() {
        if (initialized) {
            return;
        }
        try {
            TopologyDiscoveryOptions options;
            options.no_remote_discovery = false;
            auto [descriptor, chip_map] = TopologyDiscovery::discover(options);
            cluster_descriptor = std::move(descriptor);
            chips = std::move(chip_map);
            initialized = true;
        } catch (...) {
            initialized = false;
        }
    }
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
    dev.used_kernel = region->total_kernel_allocated.load(std::memory_order_relaxed);

    // Read per-process data
    dev.processes.clear();
    for (uint32_t i = 0; i < region->num_active_processes && i < SHMDeviceMemoryRegion::MAX_PROCESSES; i++) {
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
            proc.kernel_allocated = proc_entry.kernel_allocated;
            dev.processes.push_back(proc);
        }
    }

    munmap(region, sizeof(SHMDeviceMemoryRegion));
    close(fd);
    dev.has_shm = true;
    return true;
}

// Set memory sizes from SOC descriptor
static void set_memory_sizes(Device& dev, Chip* chip) {
    bool extracted = false;
    if (chip) {
        try {
            auto& soc_desc = chip->get_soc_descriptor();
            size_t num_dram_channels = soc_desc.get_num_dram_channels();
            uint32_t l1_size_per_core = soc_desc.worker_l1_size;
            auto tensix_grid = soc_desc.get_grid_size(tt::CoreType::TENSIX);
            uint64_t dram_size_per_channel = soc_desc.dram_bank_size;

            dev.total_dram = (uint64_t)num_dram_channels * dram_size_per_channel;
            dev.total_l1 = (uint64_t)l1_size_per_core * tensix_grid.x * tensix_grid.y;
            extracted = true;
        } catch (...) {
        }
    }

    if (!extracted) {
        if (dev.arch_name == "Wormhole_B0") {
            dev.total_dram = 12ULL * 1024 * 1024 * 1024;
            dev.total_l1 = 93ULL * 1024 * 1024;
        } else if (dev.arch_name == "Grayskull") {
            dev.total_dram = 8ULL * 1024 * 1024 * 1024;
            dev.total_l1 = 120ULL * 1024 * 1024;
        } else if (dev.arch_name == "Blackhole") {
            dev.total_dram = 16ULL * 1024 * 1024 * 1024;
            dev.total_l1 = 100ULL * 1024 * 1024;
        }
    }
}

// Query telemetry
static bool query_telemetry(Device& dev) {
    auto chip_it = g_topology_cache.chips.find(dev.chip_id);
    if (chip_it == g_topology_cache.chips.end()) {
        dev.telemetry.status = "No chip";
        dev.telemetry.available = false;
        return false;
    }

    Chip* chip = chip_it->second.get();
    if (!chip) {
        dev.telemetry.status = "No chip";
        dev.telemetry.available = false;
        return false;
    }

    try {
        TTDevice* tt_device = chip->get_tt_device();
        if (!tt_device) {
            dev.telemetry.status = "No device";
            dev.telemetry.available = false;
            return false;
        }

        auto firmware_info = tt_device->get_firmware_info_provider();
        if (!firmware_info) {
            dev.telemetry.status = "No FW info";
            dev.telemetry.available = false;
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
        return true;

    } catch (const std::exception& e) {
        dev.telemetry.status = "Error";
        dev.telemetry.available = false;
        return false;
    }
}

// Public API implementations
std::vector<Device> enumerate_devices() {
    std::vector<Device> devices;

    if (!g_topology_cache.initialized) {
        g_topology_cache.initialize();
    }

    if (g_topology_cache.initialized && !g_topology_cache.chips.empty()) {
        for (const auto& [chip_id, chip] : g_topology_cache.chips) {
            Device dev;
            dev.chip_id = chip_id;
            dev.arch_name = "Unknown";

            try {
                dev.is_remote = !chip->is_mmio_capable();
            } catch (...) {
                dev.is_remote = false;
            }

            try {
                TTDevice* tt_device = chip->get_tt_device();
                if (tt_device) {
                    dev.arch_name = get_arch_name(tt_device->get_arch());
                    set_memory_sizes(dev, chip.get());

                    uint64_t board_serial = tt_device->get_board_id();
                    const auto& chip_info = chip->get_chip_info();
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
                        region->total_kernel_allocated.fetch_sub(proc.kernel_allocated, std::memory_order_relaxed);
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
