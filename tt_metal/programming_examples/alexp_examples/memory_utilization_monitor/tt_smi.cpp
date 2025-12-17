// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

//
// tt_smi - Simple System Management Interface for Tenstorrent devices
//
// A lightweight monitoring tool that displays:
//   - Device telemetry (temperature, power, clocks) via UMD
//   - Memory utilization (DRAM, L1, etc.) via shared memory tracking
//
// Usage:
//   ./tt_smi              # Single snapshot
//   ./tt_smi -w           # Watch mode (updates every second)
//   ./tt_smi -w -r 500    # Watch mode (updates every 500ms)
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <utility>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>
#include <filesystem>

// TT-UMD includes for telemetry
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/topology/topology_discovery.hpp"
#include "umd/device/chip/chip.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/firmware/firmware_info_provider.hpp"

namespace fs = std::filesystem;
using namespace tt::umd;

// ANSI colors
namespace Color {
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* RED = "\033[31m";
const char* CYAN = "\033[36m";
const char* WHITE = "\033[37m";
const char* BLUE = "\033[34m";
}  // namespace Color

// Shared memory structure (must match memory_stats_shm.hpp)
struct SHMDeviceMemoryRegion {
    uint32_t version;
    uint32_t num_active_processes;
    uint64_t last_update_timestamp;
    std::atomic<uint32_t> reference_count;

    // Physical chip identification (for proper device correlation)
    uint64_t board_serial;  // Unique board serial number from UMD (0 if not available)
    uint64_t asic_id;       // ASIC ID for additional disambiguation (0 if not available)
    int32_t device_id;      // Logical device ID used for SHM file naming (backwards compat)

    std::atomic<uint64_t> total_dram_allocated;
    std::atomic<uint64_t> total_l1_allocated;
    std::atomic<uint64_t> total_l1_small_allocated;
    std::atomic<uint64_t> total_trace_allocated;
    std::atomic<uint64_t> total_cb_allocated;
    std::atomic<uint64_t> total_kernel_allocated;

    // Per-chip statistics (for tracking remote devices through gateway)
    static constexpr size_t MAX_CHIPS_PER_DEVICE = 16;
    struct ChipStats {
        uint32_t chip_id;    // Metal chip ID (0 = unused slot)
        uint32_t is_remote;  // 1 if remote chip, 0 if local
        std::atomic<uint64_t> dram_allocated;
        std::atomic<uint64_t> l1_allocated;
        std::atomic<uint64_t> l1_small_allocated;
        std::atomic<uint64_t> trace_allocated;
        std::atomic<uint64_t> cb_allocated;
        std::atomic<uint64_t> kernel_allocated;
    } chip_stats[MAX_CHIPS_PER_DEVICE];

    // Per-process tracking (up to 64 processes)
    // NOTE: These are NOT atomic (only aggregated totals are atomic)
    static constexpr size_t MAX_PROCESSES = 64;
    struct ProcessStats {
        pid_t pid;                       // Process ID (0 = unused slot)
        uint64_t dram_allocated;         // DRAM allocated by this process
        uint64_t l1_allocated;           // L1 allocated by this process
        uint64_t l1_small_allocated;     // L1_SMALL allocated by this process
        uint64_t trace_allocated;        // TRACE allocated by this process
        uint64_t cb_allocated;           // CB allocated by this process
        uint64_t kernel_allocated;       // KERNEL allocated by this process
        uint64_t last_update_timestamp;  // Last update from this process
        char process_name[64];           // Process name for debugging
    } processes[MAX_PROCESSES];

    // Padding to ensure cache line alignment
    uint8_t padding[64];
};

// Per-process memory info
struct ProcessMemoryInfo {
    pid_t pid;
    std::string name;
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
    uint64_t cb_allocated;
    uint64_t kernel_allocated;
};

// Device info
struct DeviceInfo {
    int device_id;                   // Kernel device ordinal for SHM file lookup (-1 if not found/remote)
    uint64_t chip_id_composite = 0;  // Composite ID from topology discovery (for chip_map lookup)
    uint64_t board_serial = 0;       // Physical board serial number from UMD (for correlation)
    uint64_t asic_id = 0;            // ASIC ID for display (unique per chip)
    uint8_t asic_location = 0;       // ASIC location on board (0, 1, etc.) - may be composite for Galaxy
    uint32_t tray_id = 0;            // Tray ID for Galaxy systems (0 for non-Galaxy)
    uint32_t chip_in_tray = 0;       // Chip within tray for Galaxy (0 for non-Galaxy)
    tt::BoardType board_type = tt::BoardType::UNKNOWN;  // Board type from UMD
    std::string arch_name;

    // Telemetry (cached, with age tracking)
    float temperature = -1.0f;
    float power = -1.0f;
    uint32_t aiclk_mhz = 0;

    // Telemetry metadata
    std::chrono::steady_clock::time_point last_telemetry_update;
    std::string telemetry_status = "Initializing";  // OK, Offline, Busy, etc.
    bool telemetry_failed = false;
    std::chrono::steady_clock::time_point last_telemetry_attempt;
    std::string error_type = "";  // chip_offline, eth_busy, other
    bool is_remote = false;       // True if device is remote (Ethernet-connected)

    // Memory stats from SHM
    uint64_t total_dram = 0;
    uint64_t used_dram = 0;
    uint64_t total_l1 = 0;
    uint64_t used_l1 = 0;
    uint64_t used_l1_small = 0;
    uint64_t used_trace = 0;
    uint64_t used_cb = 0;
    uint64_t used_kernel = 0;

    // Per-process breakdown
    std::vector<ProcessMemoryInfo> processes;

    // SHM state
    void* shm_region = nullptr;
    int shm_fd = -1;
    bool has_shm = false;
};

// Global cache for discovered topology and chip objects
// This is initialized once and reused across watch mode iterations
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
            options.no_remote_discovery = false;  // Enable remote device discovery

            auto [descriptor, chip_map] = TopologyDiscovery::discover(options);

            cluster_descriptor = std::move(descriptor);
            chips = std::move(chip_map);
            initialized = true;

        } catch (const std::exception& e) {
            // Topology discovery failed - will fall back to simple enumeration
            initialized = false;
        }
    }

    void reset() {
        chips.clear();
        cluster_descriptor.reset();
        initialized = false;
    }
};

static TopologyCache g_topology_cache;

// Format bytes with units
std::string format_bytes(uint64_t bytes) {
    if (bytes == 0) {
        return "0 B";
    }

    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double value = bytes;

    while (value >= 1024.0 && unit_idx < 4) {
        value /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << value << " " << units[unit_idx];
    return oss.str();
}

// Get architecture name
std::string get_arch_name(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "Grayskull";
        case tt::ARCH::WORMHOLE_B0: return "Wormhole_B0";
        case tt::ARCH::BLACKHOLE: return "Blackhole";
        default: return "Unknown";
    }
}

// Connect to shared memory for a device
bool connect_to_shm(DeviceInfo& dev) {
    if (dev.has_shm) {
        return true;  // Already connected
    }

    // Composite asic_id naming: /tt_device_<asic_id>_memory
    // Use dev.asic_id directly (already computed with proper encoding for Galaxy systems)
    // For Galaxy: asic_id = (board_id << 8) | ((tray_id << 4) | chip_in_tray)
    // For N300:   asic_id = (board_id << 8) | asic_location
    uint64_t asic_id = dev.asic_id;  // Use pre-computed asic_id
    std::string shm_name = "/tt_device_" + std::to_string(asic_id) + "_memory";

    dev.shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (dev.shm_fd < 0) {
        return false;
    }

    dev.shm_region = mmap(nullptr, sizeof(SHMDeviceMemoryRegion), PROT_READ, MAP_SHARED, dev.shm_fd, 0);
    if (dev.shm_region == MAP_FAILED) {
        close(dev.shm_fd);
        dev.shm_fd = -1;
        return false;
    }

    dev.has_shm = true;
    return true;
}

// Check if a process is still alive
bool is_process_alive(pid_t pid) {
    if (pid <= 0) {
        return false;
    }
    // Use kill(pid, 0) to check if process exists
    // Returns 0 if process exists, -1 if not (errno == ESRCH)
    return (kill(pid, 0) == 0);
}

// Read memory stats from SHM
bool read_memory_from_shm(DeviceInfo& dev, uint32_t target_chip_id = 0xFFFFFFFF) {
    if (!connect_to_shm(dev)) {
        return false;
    }

    auto* region = static_cast<SHMDeviceMemoryRegion*>(dev.shm_region);

    // NEW: Check if per-chip stats are available for this specific chip_id
    // This enables remote device tracking through gateway SHM
    bool found_chip_stats = false;
    if (target_chip_id != 0xFFFFFFFF) {
        // Look for per-chip entry matching target_chip_id
        for (size_t i = 0; i < SHMDeviceMemoryRegion::MAX_CHIPS_PER_DEVICE; i++) {
            if (region->chip_stats[i].chip_id == target_chip_id) {
                // Found it! Read per-chip stats
                dev.used_dram = region->chip_stats[i].dram_allocated.load(std::memory_order_relaxed);
                dev.used_l1 = region->chip_stats[i].l1_allocated.load(std::memory_order_relaxed);
                dev.used_l1_small = region->chip_stats[i].l1_small_allocated.load(std::memory_order_relaxed);
                dev.used_trace = region->chip_stats[i].trace_allocated.load(std::memory_order_relaxed);
                dev.used_cb = region->chip_stats[i].cb_allocated.load(std::memory_order_relaxed);
                dev.used_kernel = region->chip_stats[i].kernel_allocated.load(std::memory_order_relaxed);
                found_chip_stats = true;
                break;
            }
        }
    }

    // Fallback: Read aggregated total atomic values (backward compatibility)
    if (!found_chip_stats) {
        dev.used_dram = region->total_dram_allocated.load(std::memory_order_relaxed);
        dev.used_l1 = region->total_l1_allocated.load(std::memory_order_relaxed);
        dev.used_l1_small = region->total_l1_small_allocated.load(std::memory_order_relaxed);
        dev.used_trace = region->total_trace_allocated.load(std::memory_order_relaxed);
        dev.used_cb = region->total_cb_allocated.load(std::memory_order_relaxed);
        dev.used_kernel = region->total_kernel_allocated.load(std::memory_order_relaxed);
    }

    // Read per-process data (filter out dead processes)
    dev.processes.clear();
    uint32_t num_processes = region->num_active_processes;
    for (uint32_t i = 0; i < num_processes && i < SHMDeviceMemoryRegion::MAX_PROCESSES; i++) {
        auto& proc_entry = region->processes[i];
        pid_t pid = proc_entry.pid;

        // Only show processes that are still alive
        if (pid > 0 && is_process_alive(pid)) {
            ProcessMemoryInfo proc;
            proc.pid = pid;
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

    return true;
}

// Cleanup SHM connection
void cleanup_shm(DeviceInfo& dev) {
    if (dev.shm_region && dev.shm_region != MAP_FAILED) {
        munmap(dev.shm_region, sizeof(SHMDeviceMemoryRegion));
        dev.shm_region = nullptr;
    }
    if (dev.shm_fd >= 0) {
        close(dev.shm_fd);
        dev.shm_fd = -1;
    }
    dev.has_shm = false;
}

// Clean up dead processes from SHM (requires write access)
// Returns number of dead processes cleaned up
int cleanup_dead_processes_in_shm(DeviceInfo& dev) {
    if (!dev.has_shm || !dev.shm_region) {
        return 0;
    }

    // Re-open SHM with write access
    uint64_t asic_id = dev.asic_id;
    std::string shm_name = "/tt_device_" + std::to_string(asic_id) + "_memory";

    int fd_write = shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (fd_write < 0) {
        return 0;  // Can't open for write
    }

    auto* region_write = static_cast<SHMDeviceMemoryRegion*>(
        mmap(nullptr, sizeof(SHMDeviceMemoryRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd_write, 0));

    if (region_write == MAP_FAILED) {
        close(fd_write);
        return 0;
    }

    int cleaned_count = 0;

    // Check each process entry
    for (uint32_t i = 0; i < SHMDeviceMemoryRegion::MAX_PROCESSES; i++) {
        pid_t pid = region_write->processes[i].pid;

        if (pid > 0 && !is_process_alive(pid)) {
            // Process is dead - zero out its entry
            memset(&region_write->processes[i], 0, sizeof(SHMDeviceMemoryRegion::ProcessStats));
            cleaned_count++;
        }
    }

    // If we cleaned any processes, recompute num_active_processes
    if (cleaned_count > 0) {
        uint32_t active_count = 0;
        for (uint32_t i = 0; i < SHMDeviceMemoryRegion::MAX_PROCESSES; i++) {
            if (region_write->processes[i].pid > 0) {
                active_count++;
            }
        }
        region_write->num_active_processes = active_count;
    }

    munmap(region_write, sizeof(SHMDeviceMemoryRegion));
    close(fd_write);

    return cleaned_count;
}

// Scan /dev/shm/ for tt_device_*_memory files and parse asic_id from filenames
// Composite asic_id naming: /tt_device_<asic_id>_memory
// Returns: map of asic_id -> fd
std::map<uint64_t, int> scan_shm_files() {
    std::map<uint64_t, int> result;  // asic_id -> fd

    DIR* dir = opendir("/dev/shm");
    if (!dir) {
        return result;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;

        // Match pattern: tt_device_<asic_id>_memory
        if (filename.rfind("tt_device_", 0) == 0 && filename.find("_memory") != std::string::npos) {
            // Parse: tt_device_469504_memory -> asic_id=469504
            size_t first_underscore = 10;  // strlen("tt_device_")
            size_t second_underscore = filename.find('_', first_underscore);

            if (second_underscore != std::string::npos) {
                try {
                    uint64_t asic_id =
                        std::stoull(filename.substr(first_underscore, second_underscore - first_underscore));

                    // Open SHM to verify it's valid
                    std::string shm_name = "/" + filename;
                    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
                    if (fd >= 0) {
                        result[asic_id] = fd;
                    }
                } catch (...) {
                    // Parsing failed, skip this file
                }
            }
        }
    }

    closedir(dir);
    return result;
}

// Legacy function - no longer used but kept for compatibility
std::map<uint64_t, int> build_board_serial_to_device_id_map() {
    std::map<uint64_t, int> mapping;
    // This function is deprecated - SHM files now use physical chip IDs
    return mapping;
}

// Query telemetry from discovered Chip with resilient error handling (like tt-exalens)
// Uses the cached Chip objects from TopologyDiscovery::discover() which properly
// configures Ethernet routing for remote devices
bool query_telemetry_resilient(int device_id, DeviceInfo& dev, bool force_retry = false) {
    auto now = std::chrono::steady_clock::now();

    // Check if we should skip this attempt (cooldown after failure)
    if (dev.telemetry_failed && !force_retry) {
        auto time_since_attempt =
            std::chrono::duration_cast<std::chrono::seconds>(now - dev.last_telemetry_attempt).count();

        // Adaptive retry intervals (like tt-exalens):
        // - Chip offline: 2s (fast retry after reset)
        // - ETH busy: 15s (avoid hammering during inference)
        // - Other errors: 10s
        int cooldown_seconds = 10;  // default
        if (dev.error_type == "chip_offline") {
            cooldown_seconds = 2;
        } else if (dev.error_type == "eth_busy") {
            cooldown_seconds = 15;
        }

        if (time_since_attempt < cooldown_seconds) {
            // Still in cooldown - keep showing cached telemetry with age
            return false;
        }
    }

    dev.last_telemetry_attempt = now;

    // Try to find the chip in our topology cache using the composite ID
    auto chip_it = g_topology_cache.chips.find(dev.chip_id_composite);
    if (chip_it == g_topology_cache.chips.end()) {
        dev.telemetry_failed = true;
        dev.error_type = "other";
        dev.telemetry_status = "Not in topology";
        return false;
    }

    Chip* chip = chip_it->second.get();
    if (!chip) {
        dev.telemetry_failed = true;
        dev.error_type = "other";
        dev.telemetry_status = "Invalid chip";
        return false;
    }

    try {
        // Get the TTDevice from the Chip (works for both local and remote!)
        TTDevice* tt_device = chip->get_tt_device();
        if (!tt_device) {
            dev.telemetry_failed = true;
            dev.error_type = "other";
            dev.telemetry_status = "No TTDevice";
            return false;
        }

        // Read telemetry from TTDevice (Ethernet routing is handled by topology discovery)
        auto firmware_info = tt_device->get_firmware_info_provider();
        if (!firmware_info) {
            dev.telemetry_failed = true;
            dev.error_type = "other";
            dev.telemetry_status = "No firmware info";
            return false;
        }

        // Read telemetry
        double temp = firmware_info->get_asic_temperature();
        if (temp >= -50.0 && temp <= 100.0) {
            dev.temperature = temp;
        }

        auto tdp = firmware_info->get_tdp();
        if (tdp.has_value() && tdp.value() > 0 && tdp.value() <= 500) {
            dev.power = tdp.value();
        }

        auto aiclk = firmware_info->get_aiclk();
        if (aiclk.has_value() && aiclk.value() > 0 && aiclk.value() <= 1500) {
            dev.aiclk_mhz = aiclk.value();
        }

        // Success! Update status and clear failure flag
        dev.last_telemetry_update = now;
        dev.telemetry_failed = false;
        dev.error_type = "";
        dev.telemetry_status = "OK";
        return true;

    } catch (const std::exception& e) {
        // Device error - classify error type (like tt-exalens)
        dev.telemetry_failed = true;
        std::string error = e.what();
        std::string error_lower = error;
        std::transform(error_lower.begin(), error_lower.end(), error_lower.begin(), ::tolower);

        // Classify error type
        if (error_lower.find("0xffffffff") != std::string::npos || error_lower.find("pcie") != std::string::npos ||
            error_lower.find("reset") != std::string::npos) {
            dev.error_type = "chip_offline";
            dev.telemetry_status = "Chip offline";
        } else if (
            dev.is_remote &&
            (error_lower.find("ethernet") != std::string::npos || error_lower.find("timeout") != std::string::npos ||
             error_lower.find("configure") != std::string::npos || error_lower.find("failed") != std::string::npos)) {
            dev.error_type = "eth_busy";
            dev.telemetry_status = "ETH busy";
        } else {
            dev.error_type = "other";
            std::string short_error = error.substr(0, 15);
            dev.telemetry_status = short_error;
        }

        return false;
    }
}

// Parallel telemetry update with timeout (like tt-exalens)
// Key features for avoiding Ethernet conflicts during inference:
// 1. Each device polled in separate thread (one stuck device doesn't block others)
// 2. 1s timeout per device (prevents infinite hangs on ETH busy)
// 3. Adaptive retry intervals: 2s for chip offline, 15s for ETH busy (remote devices)
// 4. Cached telemetry with age tracking (shows last good values during failures)
// 5. Remote devices access telemetry through Ethernet fabric (may conflict with inference)
void update_all_telemetry_parallel(std::vector<DeviceInfo>& devices, bool shm_only_mode) {
    if (shm_only_mode) {
        return;  // Skip telemetry in SHM-only mode
    }

    std::vector<std::future<void>> futures;
    std::mutex update_mutex;

    for (auto& dev : devices) {
        // Launch async telemetry read with timeout (1s like tt-exalens)
        // Both local (PCIe) and remote (Ethernet) devices supported
        futures.push_back(std::async(std::launch::async, [&dev, &update_mutex]() {
            // Each device gets its own thread with 1s timeout
            auto future = std::async(std::launch::async, [&dev]() { query_telemetry_resilient(dev.device_id, dev); });

            // Wait with 1s timeout (prevents hanging on Ethernet conflicts)
            if (future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout) {
                std::lock_guard<std::mutex> lock(update_mutex);
                dev.telemetry_failed = true;
                // Remote devices timing out are likely Ethernet busy
                dev.error_type = dev.is_remote ? "eth_busy" : "other";
                dev.telemetry_status = "Timeout";
            }
        }));
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            // Ignore thread exceptions
        }
    }
}

// Get total memory sizes for architecture
void set_memory_sizes(DeviceInfo& dev) {
    // These are typical values - adjust based on actual chip specs
    if (dev.arch_name == "Wormhole_B0") {
        dev.total_dram = 12ULL * 1024 * 1024 * 1024;  // 12 GB
        dev.total_l1 = 1536ULL * 1024 * 1024;         // 1.5 GB total L1
    } else if (dev.arch_name == "Grayskull") {
        dev.total_dram = 8ULL * 1024 * 1024 * 1024;  // 8 GB
        dev.total_l1 = 1024ULL * 1024 * 1024;        // 1 GB total L1
    } else if (dev.arch_name == "Blackhole") {
        dev.total_dram = 16ULL * 1024 * 1024 * 1024;  // 16 GB
        dev.total_l1 = 2048ULL * 1024 * 1024;         // 2 GB total L1
    }
}

// Print header
void print_header() {
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    char time_buf[100];
    strftime(time_buf, sizeof(time_buf), "%a %b %d %H:%M:%S %Y", localtime(&now_t));

    std::cout << Color::BOLD << Color::CYAN
              << "+================================================================================+\n"
              << "| " << Color::WHITE << "tt-smi" << Color::CYAN << " - Tenstorrent System Management Interface"
              << std::string(29, ' ') << time_buf << " |\n"
              << "+================================================================================+" << Color::RESET
              << "\n\n";
}

// Print device table
void print_devices(const std::vector<DeviceInfo>& devices) {
    std::cout << Color::BOLD << std::left << std::setw(12) << "ID" << std::setw(14) << "Arch" << std::setw(10) << "Temp"
              << std::setw(10) << "Power" << std::setw(12) << "AICLK" << std::setw(20) << "DRAM Usage" << std::setw(20)
              << "L1 Usage" << std::setw(15) << "Status" << Color::RESET << "\n";

    std::cout << std::string(110, '-') << "\n";

    auto now = std::chrono::steady_clock::now();

    for (const auto& dev : devices) {
        // Display chip identifier based on board type
        // For Galaxy (UBB): "T<tray>:N<chip>" (e.g., "T1:N5", "T4:N8")
        // For N300/others: "<board_hex>:<asic_location>[R]" (e.g., "1834:0", "1919:1R")
        char id_str[16];

        if (dev.board_type == tt::BoardType::UBB && dev.tray_id > 0) {
            // Galaxy system: display as T<tray>:N<chip>
            snprintf(id_str, sizeof(id_str), "T%u:N%u", dev.tray_id, dev.chip_in_tray);
        } else {
            // N300 or other boards: display true unique ASIC ID from chip
            // The chip_id_composite is the 64-bit unique ASIC ID read from chip hardware
            if (dev.chip_id_composite != 0) {
                // Use the full unique ASIC ID from topology discovery
                if (dev.is_remote) {
                    snprintf(id_str, sizeof(id_str), "%llxR", (unsigned long long)dev.chip_id_composite);
                } else {
                    snprintf(id_str, sizeof(id_str), "%llx", (unsigned long long)dev.chip_id_composite);
                }
            } else {
                // Fallback: use board_serial + location if chip_id_composite not available
                uint32_t board_short = (dev.board_serial & 0xFFFFFFFF);
                if (dev.is_remote) {
                    snprintf(id_str, sizeof(id_str), "%x:%dR", board_short, dev.asic_location);
                } else {
                    snprintf(id_str, sizeof(id_str), "%x:%d", board_short, dev.asic_location);
                }
            }
        }

        std::cout << std::left << std::setw(12) << id_str;
        std::cout << std::setw(14) << dev.arch_name;

        // Calculate age of telemetry data
        auto age_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - dev.last_telemetry_update).count();
        bool is_stale = (age_seconds > 10);

        // Temperature (dim if stale)
        if (dev.temperature >= 0 && !is_stale) {
            std::cout << std::setw(10) << (std::to_string(static_cast<int>(dev.temperature)) + "°C");
        } else if (dev.temperature >= 0 && is_stale) {
            std::cout << Color::YELLOW << std::setw(10) << "---" << Color::RESET;  // Stale data
        } else {
            std::cout << Color::YELLOW << std::setw(10) << "N/A" << Color::RESET;
        }

        // Power (dim if stale)
        if (dev.power >= 0 && !is_stale) {
            std::cout << std::setw(10) << (std::to_string(static_cast<int>(dev.power)) + "W");
        } else if (dev.power >= 0 && is_stale) {
            std::cout << Color::YELLOW << std::setw(10) << "---" << Color::RESET;  // Stale data
        } else {
            std::cout << Color::YELLOW << std::setw(10) << "N/A" << Color::RESET;
        }

        // AICLK (dim if stale)
        if (dev.aiclk_mhz > 0 && !is_stale) {
            std::cout << std::setw(12) << (std::to_string(dev.aiclk_mhz) + " MHz");
        } else if (dev.aiclk_mhz > 0 && is_stale) {
            std::cout << Color::YELLOW << std::setw(12) << "---" << Color::RESET;  // Stale data
        } else {
            std::cout << Color::YELLOW << std::setw(12) << "N/A" << Color::RESET;
        }

        // DRAM usage
        if (dev.has_shm) {
            double usage_pct = (dev.total_dram > 0) ? (100.0 * dev.used_dram / dev.total_dram) : 0.0;

            if (dev.used_dram > 0) {
                std::cout << Color::GREEN;
            }
            std::cout << format_bytes(dev.used_dram) << " / " << format_bytes(dev.total_dram) << " (" << std::fixed
                      << std::setprecision(1) << usage_pct << "%)";
            if (dev.used_dram > 0) {
                std::cout << Color::RESET;
            }
        } else if (dev.total_dram > 0) {
            // No SHM - show 0 B (device not in use)
            std::cout << "0 B / " << format_bytes(dev.total_dram) << " (0.0%)";
        } else {
            std::cout << Color::YELLOW << "No data" << Color::RESET;
        }

        std::cout << "  ";

        // L1 usage
        if (dev.has_shm) {
            uint64_t total_l1_used = dev.used_l1 + dev.used_l1_small;
            double usage_pct = (dev.total_l1 > 0) ? (100.0 * total_l1_used / dev.total_l1) : 0.0;

            if (total_l1_used > 0) {
                std::cout << Color::GREEN;
            }
            std::cout << format_bytes(total_l1_used) << " / " << format_bytes(dev.total_l1) << " (" << std::fixed
                      << std::setprecision(1) << usage_pct << "%)";
            if (total_l1_used > 0) {
                std::cout << Color::RESET;
            }
        } else if (dev.total_l1 > 0) {
            // No SHM - show 0 B (device not in use)
            std::cout << std::setw(20) << ("0 B / " + format_bytes(dev.total_l1) + " (0.0%)");
        } else {
            std::cout << Color::YELLOW << std::setw(20) << "No data" << Color::RESET;
        }

        std::cout << "  ";

        // Status column (shows telemetry health)
        if (dev.telemetry_status == "OK") {
            if (is_stale) {
                auto age_s = age_seconds;
                std::cout << Color::YELLOW << std::setw(15) << ("Stale (" + std::to_string(age_s) + "s)")
                          << Color::RESET;
            } else {
                std::cout << Color::GREEN << std::setw(15) << "OK" << Color::RESET;
            }
        } else {
            // Show error status
            std::string status = dev.telemetry_status;
            if (status.length() > 14) {
                status = status.substr(0, 11) + "...";
            }
            std::cout << Color::YELLOW << std::setw(15) << status << Color::RESET;
        }

        std::cout << "\n";
    }

    std::cout << "\n";
}

// Print per-process memory usage
void print_process_table(const std::vector<DeviceInfo>& devices) {
    bool any_processes = false;
    bool any_memory = false;

    for (const auto& dev : devices) {
        if (!dev.processes.empty()) {
            any_processes = true;
        }
        if (dev.used_dram > 0 || dev.used_l1 > 0) {
            any_memory = true;
        }
    }

    if (!any_processes && any_memory) {
        std::cout << "\n"
                  << Color::YELLOW << "⚠  Per-PID memory tracking disabled (aggregated stats only)\n"
                  << Color::RESET;
        std::cout << Color::CYAN << "   Enable with: export TT_METAL_SHM_STATS_PER_PID=1\n" << Color::RESET;
        std::cout << Color::CYAN << "   Then restart your TT-Metal program\n" << Color::RESET;
        return;
    }

    if (!any_processes) {
        return;  // No processes and no memory - nothing to show
    }

    std::cout << "\n" << Color::BOLD << "Per-Process Memory Usage:\n" << Color::RESET;
    std::cout << std::string(100, '-') << "\n";

    std::cout << Color::BOLD << std::left << std::setw(12) << "Dev" << std::setw(8) << "PID" << std::setw(20)
              << "Process" << std::setw(14) << "DRAM" << std::setw(14) << "L1" << std::setw(14) << "L1 Small"
              << std::setw(12) << "Trace" << Color::RESET << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& dev : devices) {
        for (const auto& proc : dev.processes) {
            // Show same ID format as main table (use unique ASIC ID)
            char id_str[16];
            if (dev.board_type == tt::BoardType::UBB && dev.tray_id > 0) {
                snprintf(id_str, sizeof(id_str), "T%u:N%u", dev.tray_id, dev.chip_in_tray);
            } else if (dev.chip_id_composite != 0) {
                if (dev.is_remote) {
                    snprintf(id_str, sizeof(id_str), "%llxR", (unsigned long long)dev.chip_id_composite);
                } else {
                    snprintf(id_str, sizeof(id_str), "%llx", (unsigned long long)dev.chip_id_composite);
                }
            } else {
                // Fallback
                uint32_t board_short = (dev.board_serial & 0xFFFFFFFF);
                if (dev.is_remote) {
                    snprintf(id_str, sizeof(id_str), "%x:%dR", board_short, dev.asic_location);
                } else {
                    snprintf(id_str, sizeof(id_str), "%x:%d", board_short, dev.asic_location);
                }
            }
            std::cout << std::left << std::setw(12) << id_str;
            std::cout << std::setw(8) << proc.pid;

            // Truncate process name if too long
            std::string proc_name = proc.name;
            if (proc_name.length() > 18) {
                proc_name = proc_name.substr(0, 15) + "...";
            }
            std::cout << std::setw(20) << proc_name;

            std::cout << Color::GREEN << std::setw(14) << format_bytes(proc.dram_allocated);
            std::cout << std::setw(14) << format_bytes(proc.l1_allocated);
            std::cout << std::setw(14) << format_bytes(proc.l1_small_allocated);
            std::cout << std::setw(12) << format_bytes(proc.trace_allocated);
            std::cout << Color::RESET << "\n";
        }
    }

    std::cout << "\n";
}

// Print detailed memory breakdown
void print_memory_details(const std::vector<DeviceInfo>& devices) {
    bool any_tracking = false;
    for (const auto& dev : devices) {
        if (dev.has_shm) {
            any_tracking = true;
            break;
        }
    }

    if (!any_tracking) {
        std::cout << Color::YELLOW << "⚠  No memory tracking data available\n" << Color::RESET;
        std::cout << Color::CYAN << "   Enable tracking with: export TT_METAL_ENABLE_SHM_TRACKING=1\n" << Color::RESET;
        std::cout << Color::CYAN << "   Then run a TT-Metal program to populate stats\n" << Color::RESET;
        return;
    }

    std::cout << Color::BOLD << "Memory Details by Type:\n" << Color::RESET;
    std::cout << std::string(95, '-') << "\n";

    for (const auto& dev : devices) {
        if (!dev.has_shm) {
            continue;
        }

        std::string id_label = std::to_string(dev.asic_id);
        if (dev.is_remote) {
            id_label += "R";
        }
        std::cout << Color::BOLD << "Device " << id_label << ":\n" << Color::RESET;
        std::cout << "  DRAM:      " << std::setw(12) << format_bytes(dev.used_dram) << "\n";
        std::cout << "  L1:        " << std::setw(12) << format_bytes(dev.used_l1) << "\n";
        std::cout << "  L1 Small:  " << std::setw(12) << format_bytes(dev.used_l1_small) << "\n";
        std::cout << "  Trace:     " << std::setw(12) << format_bytes(dev.used_trace) << "\n";
        std::cout << "  CB:        " << std::setw(12) << format_bytes(dev.used_cb) << "\n";
        std::cout << "  Kernel:    " << std::setw(12) << format_bytes(dev.used_kernel) << "\n";
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    bool watch_mode = false;
    int refresh_ms = 1000;
    bool show_details = false;
    bool shm_only = false;  // NEW: Skip device access, only read SHM
    bool auto_cleanup = false;  // NEW: Automatically clean up dead processes

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--watch") {
            watch_mode = true;
        } else if (arg == "-r" || arg == "--refresh") {
            if (i + 1 < argc) {
                refresh_ms = std::stoi(argv[++i]);
            }
        } else if (arg == "-d" || arg == "--details") {
            show_details = true;
        } else if (arg == "--shm-only" || arg == "--memory-only") {
            shm_only = true;
        } else if (arg == "-c" || arg == "--cleanup") {
            auto_cleanup = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -w, --watch          Watch mode (continuous updates)\n";
            std::cout << "  -r, --refresh MS     Refresh interval in milliseconds (default: 1000)\n";
            std::cout << "  -d, --details        Show detailed memory breakdown\n";
            std::cout << "  -c, --cleanup        Automatically cleanup dead processes from SHM\n";
            std::cout << "  --shm-only           Only read SHM (no device access - works during reset)\n";
            std::cout << "  -h, --help           Show this help\n";
            std::cout << "\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << "                    # One-time snapshot\n";
            std::cout << "  " << argv[0] << " -w                 # Watch mode\n";
            std::cout << "  " << argv[0] << " -w -c              # Watch mode with auto-cleanup\n";
            std::cout << "  " << argv[0] << " --cleanup          # Clean dead processes once and exit\n";
            std::cout << "\n";
            std::cout << "Note: --shm-only mode is non-invasive and won't prevent device reset\n";
            return 0;
        }
    }

    // Hide cursor in watch mode
    if (watch_mode) {
        std::cout << "\033[?25l";  // Hide cursor
    }

    try {
        // In watch mode, clear screen once at start
        if (watch_mode) {
            std::cout << "\033[2J\033[H";  // Clear screen and move to top
        }

        do {
            if (watch_mode) {
                std::cout << "\033[H";  // Move cursor to home (top-left) without clearing
                std::cout << std::flush;
            }

            print_header();

            // Enumerate devices
            std::vector<DeviceInfo> devices;

            if (shm_only) {
                // SHM-only mode: enumerate devices by scanning /dev/shm for tt_device_*_memory files
                // This is completely non-invasive and works even during device reset
                for (int i = 0; i < 256; i++) {
                    DeviceInfo dev;
                    dev.device_id = i;
                    dev.arch_name = "Unknown";  // Can't get arch without opening device

                    // Try to connect to SHM
                    if (read_memory_from_shm(dev)) {
                        // Guess memory sizes based on common configs (or leave as 0)
                        dev.total_dram = 12ULL * 1024 * 1024 * 1024;  // 12 GB typical
                        dev.total_l1 = 1536ULL * 1024 * 1024;         // 1.5 GB typical
                        dev.telemetry_status = "SHM-only mode";
                        devices.push_back(dev);
                    }
                }
            } else {
                // Normal mode: use TopologyDiscovery to enumerate ALL devices (local + remote)
                // Initialize topology cache on first run
                if (!g_topology_cache.initialized) {
                    g_topology_cache.initialize();
                }

                if (g_topology_cache.initialized && !g_topology_cache.chips.empty()) {
                    // Strategy: First scan all SHM files, then match with topology chips
                    // This allows us to see which Metal device IDs have SHM data

                    // Scan /dev/shm/ for tt_device_*_memory files
                    // Returns: map of asic_id -> fd
                    auto chip_shm_map = scan_shm_files();  // asic_id -> fd

                    // Now enumerate topology chips
                    for (const auto& [chip_id_composite, chip] : g_topology_cache.chips) {
                        DeviceInfo dev;
                        dev.chip_id_composite = chip_id_composite;
                        dev.arch_name = "Unknown";
                        dev.device_id = -1;

                        // Determine if remote (no MMIO access)
                        try {
                            dev.is_remote = !chip->is_mmio_capable();
                        } catch (...) {
                            dev.is_remote = false;
                        }

                        // Get arch and chip info
                        try {
                            TTDevice* tt_device = chip->get_tt_device();
                            if (tt_device) {
                                auto arch = tt_device->get_arch();
                                dev.arch_name = get_arch_name(arch);
                                set_memory_sizes(dev);

                                // Get the actual board_id
                                dev.board_serial = tt_device->get_board_id();
                            }

                            // Get asic_location and board_type
                            const auto& chip_info = chip->get_chip_info();
                            dev.asic_location = chip_info.asic_location;
                            dev.board_type = chip_info.board_type;

                            // For Galaxy (UBB) systems, decode tray and chip from asic_location_composite
                            // Format: bits 4-7 = tray_id, bits 0-3 = chip_in_tray
                            if (dev.board_type == tt::BoardType::UBB && tt_device && tt_device->get_pci_device()) {
                                uint16_t pci_bus = tt_device->get_pci_device()->get_device_info().pci_bus;

                                // Map PCI bus upper nibble to tray (Wormhole Galaxy convention)
                                static const std::vector<uint16_t> tray_bus_ids = {0xC0, 0x80, 0x00, 0x40};
                                uint16_t bus_upper = pci_bus & 0xF0;
                                auto tray_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_upper);

                                if (tray_it != tray_bus_ids.end()) {
                                    dev.tray_id = static_cast<uint32_t>(tray_it - tray_bus_ids.begin()) + 1;
                                    dev.chip_in_tray = pci_bus & 0x0F;

                                    // Encode for asic_id computation
                                    uint32_t asic_location_composite = (dev.tray_id << 4) | dev.chip_in_tray;
                                    dev.asic_id = (dev.board_serial << 8) | asic_location_composite;
                                } else {
                                    // Fallback if PCI bus pattern doesn't match
                                    dev.asic_id = (dev.board_serial << 8) | dev.asic_location;
                                }
                            } else {
                                // Non-Galaxy systems: use asic_location directly
                                dev.asic_id = (dev.board_serial << 8) | dev.asic_location;
                            }
                        } catch (...) {
                        }

                        // Direct asic_id lookup - globally unique identifier!
                        // Uses the computed asic_id which includes tray encoding for Galaxy
                        uint64_t asic_id = dev.asic_id;

                        if (chip_shm_map.count(asic_id)) {
                            // Found SHM for this chip!
                            dev.device_id = 0;  // Placeholder - not used for SHM lookup anymore

                            // Read memory stats from this chip's SHM
                            read_memory_from_shm(dev);
                        }
                        // else: No SHM for this chip - will show 0 B (default)

                        devices.push_back(dev);
                    }

                    // Clean up SHM file descriptors
                    for (auto& [asic_id, fd] : chip_shm_map) {
                        close(fd);
                    }

                    // Sort devices by ASIC ID (board_id + asic_location) for consistent display
                    std::sort(devices.begin(), devices.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
                        if (a.board_serial != b.board_serial) {
                            return a.board_serial < b.board_serial;
                        }
                        return a.asic_location < b.asic_location;
                    });

                    // Parallel telemetry update for ALL devices (non-blocking, with timeout)
                    // Uses cached Chip objects which have Ethernet routing configured
                    update_all_telemetry_parallel(devices, false);
                }

                // Fallback: enumerate local PCIe devices only (if cluster desc failed)
                if (devices.empty()) {
                    for (int i = 0; i < 8; i++) {
                        DeviceInfo dev;
                        dev.device_id = i;
                        dev.arch_name = "Unknown";
                        dev.is_remote = false;  // Local PCIe devices

                        // Try to read memory stats from SHM (non-blocking)
                        bool has_memory = read_memory_from_shm(dev);

                        if (has_memory) {
                            // Has SHM - assume device exists
                            dev.total_dram = 12ULL * 1024 * 1024 * 1024;
                            dev.total_l1 = 1536ULL * 1024 * 1024;
                            devices.push_back(dev);
                        }
                    }

                    // Parallel telemetry update (non-blocking, with timeout)
                    update_all_telemetry_parallel(devices, false);
                }
            }

            if (devices.empty()) {
                std::cout << Color::RED << "No Tenstorrent devices found!\n" << Color::RESET;
                return 1;
            }

            // Clean up dead processes if requested
            if (auto_cleanup) {
                int total_cleaned = 0;
                for (auto& dev : devices) {
                    if (dev.has_shm) {
                        int cleaned = cleanup_dead_processes_in_shm(dev);
                        total_cleaned += cleaned;
                    }
                }
                if (total_cleaned > 0 && !watch_mode) {
                    std::cout << Color::GREEN << "Cleaned up " << total_cleaned << " dead process(es) from SHM\n"
                              << Color::RESET;
                }
            }

            // Print device table
            print_devices(devices);

            // Print per-process memory usage
            print_process_table(devices);

            // Print detailed breakdown if requested
            if (show_details) {
                print_memory_details(devices);
            }

            // Clear to end of screen to erase any old content (prevents flicker artifacts)
            if (watch_mode) {
                std::cout << "\033[J";  // Clear from cursor to end of screen
                std::cout << std::flush;
            }

            // Cleanup
            for (auto& dev : devices) {
                cleanup_shm(dev);
            }

            if (watch_mode) {
                std::cout << Color::CYAN << "Press Ctrl+C to exit...\r" << Color::RESET;
                std::cout << std::flush;
                std::this_thread::sleep_for(std::chrono::milliseconds(refresh_ms));
            }

        } while (watch_mode);

    } catch (const std::exception& e) {
        std::cerr << Color::RED << "Error: " << e.what() << Color::RESET << "\n";
        return 1;
    }

    // Show cursor again
    if (watch_mode) {
        std::cout << "\033[?25h";  // Show cursor
    }

    // Cleanup topology cache
    g_topology_cache.reset();

    return 0;
}
