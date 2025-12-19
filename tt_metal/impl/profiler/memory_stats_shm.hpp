// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/types.h>

namespace tt::tt_metal {

// Maximum number of processes that can be tracked per device
constexpr size_t MAX_PROCESSES = 64;

// Maximum number of chips that can be tracked through a single device (gateway)
// For N300: 1 local + 1 remote = 2; for larger meshes: 1 local + up to 15 remote = 16
constexpr size_t MAX_CHIPS_PER_DEVICE = 16;

// Shared memory region layout for per-device memory statistics
// This structure is mapped into shared memory at /dev/shm/tt_device_N_memory
// NOTE: SHM files persist across runs (like UMD locks) - manual cleanup: rm /dev/shm/tt_device_*
struct DeviceMemoryRegion {
    // Header information
    uint32_t version;                       // Structure version (for compatibility)
    uint32_t num_active_processes;          // Number of processes currently tracked (in per-PID array)
    uint64_t last_update_timestamp;         // Last update time (nanoseconds since epoch)
    std::atomic<uint32_t> reference_count;  // Number of processes currently attached (always tracked)

    // Physical chip identification (for proper device correlation)
    uint64_t board_serial;  // Unique board serial number from UMD (0 if not available)
    uint64_t asic_id;       // ASIC ID for additional disambiguation (0 if not available)
    int32_t device_id;      // Logical device ID used for SHM file naming (backwards compat)

    // Aggregated device-wide statistics (updated atomically on every allocation)
    // These counters track total memory usage across ALL processes and ALL chips
    std::atomic<uint64_t> total_dram_allocated;
    std::atomic<uint64_t> total_l1_allocated;
    std::atomic<uint64_t> total_l1_small_allocated;
    std::atomic<uint64_t> total_trace_allocated;
    std::atomic<uint64_t> total_cb_allocated;      // Circular buffers
    std::atomic<uint64_t> total_kernel_allocated;  // Kernel code

    // Per-chip statistics (for tracking remote devices through gateway)
    // chip_stats[0] = gateway (local) chip allocations
    // chip_stats[1..N] = remote chip allocations accessed through this gateway
    struct ChipStats {
        uint32_t chip_id;    // Metal chip ID (0 = unused slot)
        uint32_t is_remote;  // 1 if remote chip, 0 if local (gateway)
        std::atomic<uint64_t> dram_allocated;
        std::atomic<uint64_t> l1_allocated;
        std::atomic<uint64_t> l1_small_allocated;
        std::atomic<uint64_t> trace_allocated;
        std::atomic<uint64_t> cb_allocated;
        std::atomic<uint64_t> kernel_allocated;
    } chip_stats[MAX_CHIPS_PER_DEVICE];

    // Per-process breakdown (optional, for detailed tracking)
    struct ProcessStats {
        pid_t pid;                       // Process ID (0 = unused slot)
        uint64_t dram_allocated;         // DRAM allocated by this process
        uint64_t l1_allocated;           // L1 allocated by this process
        uint64_t l1_small_allocated;     // L1_SMALL allocated by this process
        uint64_t trace_allocated;        // TRACE allocated by this process
        uint64_t cb_allocated;           // CB allocated by this process
        uint64_t kernel_allocated;       // KERNEL allocated by this process
        uint64_t last_update_timestamp;  // Last update from this process
        char process_name[64];           // Optional: process name for debugging
    } processes[MAX_PROCESSES];

    // Padding to ensure cache line alignment
    uint8_t padding[64];
} __attribute__((aligned(64)));

// Buffer types (matching tt_metal::BufferType)
enum class ShmBufferType : uint8_t {
    DRAM = 0,
    L1 = 1,
    SYSTEM_MEMORY = 2,
    L1_SMALL = 3,
    TRACE = 4,
    CB = 5,
    KERNEL = 6,
    UNKNOWN = 255
};

// Shared memory statistics provider - manages per-device memory tracking
// Each device gets its own PERSISTENT shared memory region at /dev/shm/tt_device_N_memory
// Files persist like UMD locks (/dev/shm/TT_UMD_LOCK.*) for continuous monitoring
class SharedMemoryStatsProvider {
public:
    // Create or attach to shared memory for the given device
    // If first process, initializes the region; otherwise attaches to existing
    // Uses composite asic_id for SHM naming to ensure global uniqueness
    // asic_id: Composite ID = (board_id << 8) | asic_location (globally unique)
    // device_id: Logical Metal device ID (for internal tracking only)
    explicit SharedMemoryStatsProvider(uint64_t asic_id, int device_id);

    // Destructor unmaps shared memory and closes file descriptor
    // NOTE: SHM file persists (like UMD locks) - not deleted on process exit
    ~SharedMemoryStatsProvider();

    // Prevent copying (shared memory region is unique per device)
    SharedMemoryStatsProvider(const SharedMemoryStatsProvider&) = delete;
    SharedMemoryStatsProvider& operator=(const SharedMemoryStatsProvider&) = delete;

    // Record an allocation (updates aggregated, per-chip, and optionally per-PID stats)
    // This is called on every buffer allocation (fast path - ~20-50ns)
    // chip_id: Metal chip ID where this buffer is allocated (for remote chip tracking)
    void record_allocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id = 0);

    // Record a deallocation (updates aggregated, per-chip, and optionally per-PID stats)
    // This is called on every buffer deallocation (fast path - ~20-50ns)
    // chip_id: Metal chip ID where this buffer was allocated
    void record_deallocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id = 0);

    // Get current device-wide statistics (read-only, no locks)
    struct DeviceStats {
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
        uint64_t kernel_allocated;
        uint64_t timestamp;
    };
    DeviceStats get_device_stats() const;

    // Get per-process statistics (returns empty vector if per-PID tracking disabled)
    struct ProcessInfo {
        pid_t pid;
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
        uint64_t kernel_allocated;
        uint64_t timestamp;
        std::string process_name;
    };
    std::vector<ProcessInfo> get_process_stats() const;

    // Get per-chip statistics (for remote device tracking)
    struct ChipInfo {
        uint32_t chip_id;
        bool is_remote;
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
        uint64_t kernel_allocated;
    };
    std::vector<ChipInfo> get_chip_stats() const;

    // Register a chip for tracking (called when MeshDevice includes remote chips)
    // This sets up the chip_stats entry so allocations can be attributed correctly
    void register_chip(uint32_t chip_id, bool is_remote);

    // Check if shared memory is initialized and valid
    bool is_initialized() const { return region_ != nullptr; }

    // Get device ID this provider is tracking
    int device_id() const { return device_id_; }

    // Get composite asic_id
    uint64_t asic_id() const { return asic_id_; }

    // Enable/disable per-PID tracking (default: enabled, disable with TT_METAL_SHM_TRACKING_DISABLED=1)
    void set_per_pid_tracking(bool enabled) { per_pid_tracking_enabled_ = enabled; }
    bool is_per_pid_tracking_enabled() const { return per_pid_tracking_enabled_; }

private:
    uint64_t asic_id_;               // Composite asic_id = (board_id << 8) | asic_location (for SHM naming)
    int device_id_;                  // Logical Metal device ID (for internal tracking)
    int shm_fd_;                     // Shared memory file descriptor
    DeviceMemoryRegion* region_;     // Mapped shared memory region
    bool per_pid_tracking_enabled_;  // Enable detailed per-PID tracking
    bool is_creator_;                // True if this process created the shared memory

    // Helper: Initialize shared memory region (first process only)
    void initialize_region();

    // Helper: Find or create per-PID stats entry
    DeviceMemoryRegion::ProcessStats* find_or_create_pid_entry(pid_t pid);

    // Helper: Find or create per-chip stats entry
    DeviceMemoryRegion::ChipStats* find_or_create_chip_entry(uint32_t chip_id);

    // Helper: Get current timestamp in nanoseconds
    static uint64_t current_timestamp_ns();

    // Helper: Get process name from PID
    static std::string get_process_name(pid_t pid);
};

}  // namespace tt::tt_metal
