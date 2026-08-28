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
constexpr size_t MAX_PROCESSES = 64u;

// Maximum number of chips that can be tracked through a single device (gateway)
// For N300: 1 local + 1 remote = 2; for larger meshes: 1 local + up to 15 remote = 16
constexpr size_t MAX_CHIPS_PER_DEVICE = 16u;

// Sentinel value for unused ChipStats slots. Chip ID 0 is a valid device,
// so we use UINT32_MAX (never a real chip ID) to mark empty entries.
constexpr uint32_t CHIP_STATS_UNUSED = UINT32_MAX;

// DeviceMemoryRegion structure version
// IMPORTANT: Increment this version number whenever DeviceMemoryRegion, ChipStats,
// or ProcessStats structures are modified (fields added/removed/reordered).
// Readers should check this version to ensure compatibility with the SHM layout.
// v2: asic_id now stores UMD chip_unique_id directly (matches SHM filename)
//     chip_stats sentinel is CHIP_STATS_UNUSED (UINT32_MAX), not 0
// v3: last_update_timestamp, ChipStats::chip_id and ChipStats::is_remote are atomic
// v4: init_state publishes readiness so attaching processes cannot observe a
//     half-initialized (zero-filled) region; num_active_processes is atomic;
//     a process claims its ProcessStats slot at attach time rather than on its
//     first allocation; every aggregated total and reference_count is moved by
//     DELTA (add on allocate, saturating subtract on free and on slot release),
//     so a process that dies without running its destructor has its bytes
//     subtracted by whoever reaps its slot instead of leaving ghost allocations.
//     Deltas rather than a recomputed sum-then-store: the latter drops any
//     allocation another writer records between the sum and the store.
//     Field offsets 0..(processes end) are unchanged from v3 so that a reader
//     which only understands v3 still parses the fields it knows.
constexpr uint32_t DEVICE_MEMORY_REGION_VERSION = 4;

// A region of an older layout is taken over only when no owning process is still alive.
// reference_count cannot decide it: an older writer leaks it permanently when killed, which
// would make the upgrade one-way. Requires knowing where `processes` is; see
// process_table_layout_is_known() and legacy_region_is_reclaimable().
//
// Note that a pre-v4 region cannot be recognised by init_state, which it never published: the
// field is appended last, so it reads UNINITIALIZED in any older layout. `version` is what
// separates "never written" from "written by a layout that predates init_state", and attach
// applies the live-owner test to the latter rather than treating it as a fresh region.

// Values for DeviceMemoryRegion::init_state. A freshly created SHM region is
// zero-filled, so UNINITIALIZED must be 0.
constexpr uint32_t SHM_INIT_UNINITIALIZED = 0u;
constexpr uint32_t SHM_INIT_IN_PROGRESS = 1u;
constexpr uint32_t SHM_INIT_READY = 0x52454459u;  // 'REDY'

// A ProcessStats slot whose pid reads this is mid-release: its owner is gone but its counters
// have not been cleared yet. Not claimable (claim_own_pid_entry only takes a slot whose pid is
// 0) and not a live pid (process_is_alive rejects <= 0), so it is invisible to everything but
// the releaser that put it there. Whoever wins the CAS to it owns the release, which is what
// keeps two reapers from clearing one slot twice.
constexpr pid_t SHM_SLOT_RELEASING = -1;

// Shared memory region layout for per-device memory statistics
// This structure is mapped into shared memory at /dev/shm/tt_device_*_memory
// SHM files persist across runs (like UMD locks) - manual cleanup: rm /dev/shm/tt_device_*
struct DeviceMemoryRegion {
    // Header information
    // Atomic: read by attaching processes concurrently with the initializing process's
    // write, and by external readers at any time (tt-mgmt). Layout matches uint32_t.
    std::atomic<uint32_t> version;
    // Number of live entries in `processes`. Derived, not independently counted.
    std::atomic<uint32_t> num_active_processes;
    std::atomic<uint64_t> last_update_timestamp;  // Last update time (nanoseconds since epoch)
    // Number of processes currently attached. Derived from the live `processes` entries, so a
    // SIGKILLed process is reclaimed by the next attacher instead of pinning this above zero.
    std::atomic<uint32_t> reference_count;

    // Physical chip identification (for proper device correlation)
    // SHM filename uses chip_unique_id: /dev/shm/tt_device_<chip_unique_id>_memory
    // Atomic: every attaching process rewrites these, so concurrent attaches would
    // otherwise be plain concurrent writes to the same words. Layout is unchanged.
    std::atomic<uint64_t> board_serial;  // Reserved (0), use UMD board_id APIs for board correlation
    std::atomic<uint64_t> asic_id;       // UMD chip_unique_id - globally unique, matches SHM filename
    std::atomic<uint32_t> device_id;     // Logical Metal device ID (unsigned; ChipId validated >= 0)

    // Aggregated device-wide statistics (updated atomically on every allocation)
    // These counters track total memory usage across ALL processes and ALL chips
    std::atomic<uint64_t> total_dram_allocated;
    std::atomic<uint64_t> total_l1_allocated;
    std::atomic<uint64_t> total_l1_small_allocated;
    std::atomic<uint64_t> total_trace_allocated;
    std::atomic<uint64_t> total_cb_allocated;  // Circular buffers

    // Per-chip statistics (for tracking remote devices through gateway)
    // chip_stats[0] = gateway (local) chip allocations
    // chip_stats[1..N] = remote chip allocations accessed through this gateway
    struct ChipStats {
        std::atomic<uint32_t> chip_id;    // Metal chip ID (CHIP_STATS_UNUSED = empty slot)
        std::atomic<uint32_t> is_remote;  // 1 if remote chip, 0 if local (gateway)
        std::atomic<uint64_t> dram_allocated;
        std::atomic<uint64_t> l1_allocated;
        std::atomic<uint64_t> l1_small_allocated;
        std::atomic<uint64_t> trace_allocated;
        std::atomic<uint64_t> cb_allocated;
    } chip_stats[MAX_CHIPS_PER_DEVICE];

    // Per-process breakdown (optional, for detailed tracking)
    struct ProcessStats {
        std::atomic<pid_t> pid;                       // Process ID (0 = unused slot)
        std::atomic<uint64_t> dram_allocated;         // DRAM allocated by this process
        std::atomic<uint64_t> l1_allocated;           // L1 allocated by this process
        std::atomic<uint64_t> l1_small_allocated;     // L1_SMALL allocated by this process
        std::atomic<uint64_t> trace_allocated;        // TRACE allocated by this process
        std::atomic<uint64_t> cb_allocated;           // CB allocated by this process
        std::atomic<uint64_t> last_update_timestamp;  // Last update from this process
        char process_name[64];                        // Optional: process name for debugging
    } processes[MAX_PROCESSES];

    // Publishes whether the fields above are valid; last, so every v3 field keeps its offset.
    // Without it an attacher can observe a zero-filled region, where chip_id reads 0 -- a
    // valid chip ID, defeating the CHIP_STATS_UNUSED sentinel. Release on write, acquire on read.
    std::atomic<uint32_t> init_state;
} __attribute__((aligned(64)));

// Buffer types (matching tt_metal::BufferType)
enum class ShmBufferType : uint8_t {
    DRAM = 0,
    L1 = 1,
    SYSTEM_MEMORY = 2,
    L1_SMALL = 3,
    TRACE = 4,
    CB = 5,
    UNKNOWN = 255
};

// Shared memory statistics provider - manages per-device memory tracking
// Each device gets its own PERSISTENT shared memory region at /dev/shm/tt_device_N_memory
class SharedMemoryStatsProvider {
public:
    // Create or attach to shared memory for the given device
    // If first process, initializes the region; otherwise attaches to existing
    // asic_id: UMD chip_unique_id from ClusterDescriptor (globally unique per chip)
    // device_id: Logical Metal device ID (for internal tracking only)
    // tracking_disabled: process-wide TT_METAL_SHM_TRACKING_DISABLED flag, captured at
    //                    construction time from the owning Device's MetalContext rtoptions
    //                    so the SHM provider does not need to walk MetalContext slots later
    // verbose: process-wide TT_METAL_SHM_VERBOSE flag, captured the same way
    SharedMemoryStatsProvider(uint64_t asic_id, int device_id, bool tracking_disabled, bool verbose);

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
        uint64_t timestamp;
        std::string process_name;
    };
    std::vector<ProcessInfo> get_process_stats() const;

    // Update SHM with current allocator statistics (queries allocator directly)
    // This replaces cumulative allocation/deallocation tracking with ground truth
    // Call this periodically or on-demand to sync SHM with actual allocator state
    // device: Device to query (can be nullptr to skip update)
    // pid: Process ID for per-process tracking
    void update_from_allocator(const class Device* device, pid_t pid);

    // Get per-chip statistics (for remote device tracking)
    struct ChipInfo {
        uint32_t chip_id;
        bool is_remote;
        uint64_t dram_allocated;
        uint64_t l1_allocated;
        uint64_t l1_small_allocated;
        uint64_t trace_allocated;
        uint64_t cb_allocated;
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
    uint64_t asic_id_;               // UMD chip_unique_id (for SHM naming)
    int device_id_;                  // Logical Metal device ID (for internal tracking)
    int shm_fd_;                     // Shared memory file descriptor
    DeviceMemoryRegion* region_;     // Mapped shared memory region
    bool per_pid_tracking_enabled_;  // Enable detailed per-PID tracking
    bool verbose_enabled_;           // Cached TT_METAL_SHM_VERBOSE flag (process-wide)
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

    // Helper: is this PID still alive? Used to reclaim slots left behind by
    // processes that died without running their destructor.
    static bool process_is_alive(pid_t pid);

    // May `processes` be read in a region written by this version? Gates inspecting a
    // foreign-layout region rather than trusting its leak-prone reference_count.
    static bool process_table_layout_is_known(uint32_t version);

    // Read-only; valid only when process_table_layout_is_known() for the region's version.
    bool region_has_live_process() const;

    // May this process take over a region that `found_version` wrote and reinitialize it to
    // the current layout? True only when nothing can still be using it: either the region was
    // never written (version 0, so it is zero-filled and there is nothing to lose) or its
    // layout is one we know and no live process owns a slot there.
    //
    // reference_count cannot answer this on its own -- an older writer leaks it permanently
    // when killed, which would make the upgrade one-way -- so the process table is what
    // decides, wherever we know where to find it.
    bool legacy_region_is_reclaimable(uint32_t found_version) const;

    // Helper: unmap the region and close the fd, leaving this provider disabled. Used
    // wherever attach decides it must not touch a region it cannot safely take over.
    void detach_region();

    // Helper: wait (bounded) for another process to finish initializing the region.
    // Returns false on timeout, in which case this provider disables itself rather
    // than operate on a region whose contents it cannot trust.
    bool wait_for_region_ready();

    // Helper: claim this process's ProcessStats slot. Called once at attach time
    // (not lazily on first allocation) so that attachment is discoverable by other
    // processes and reclaimable if we die.
    DeviceMemoryRegion::ProcessStats* claim_own_pid_entry(pid_t pid);

    // Helper: subtract a slot's bytes from the device-wide totals, then zero the slot and
    // drop the attached count. Every aggregate mutation in this class is a delta like this
    // one; see saturating_sub.
    // Release `slot`, which must be believed to belong to `expected_owner`. Conditional on that
    // pid: between deciding a slot is releasable and getting here it can have been released by
    // somebody else and re-claimed by a live process, and clearing that process's slot would
    // strand it -- attached, counted, and reporting nothing.
    void release_process_slot(DeviceMemoryRegion::ProcessStats& slot, pid_t expected_owner);

    // Helper: subtract without wrapping, for the unsigned shared counters.
    static void saturating_sub(std::atomic<uint64_t>& counter, uint64_t amount);
    static void saturating_sub(std::atomic<uint32_t>& counter, uint32_t amount);

    // Helper: reclaim slots whose owning process no longer exists. Their memory was
    // never released, so this is what stops a SIGKILLed run from leaving permanently
    // inflated totals behind (tt-mgmt's `smi cleanup` does not fix that case).
    // Returns the number of slots reclaimed.
    size_t reap_dead_processes();
};

}  // namespace tt::tt_metal
