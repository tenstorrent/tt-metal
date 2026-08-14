// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/memory_tracking/memory_stats_shm.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tt::tt_metal {

// Forward declarations to avoid circular dependencies
class Device;
class Allocator;

// Implementation of SharedMemoryStatsProvider

SharedMemoryStatsProvider::SharedMemoryStatsProvider(
    uint64_t asic_id, int device_id, bool tracking_disabled, bool verbose) :
    asic_id_(asic_id),
    device_id_(device_id),
    shm_fd_(-1),
    region_(nullptr),
    // Per-PID tracking is enabled by default and disabled by TT_METAL_SHM_TRACKING_DISABLED=1.
    // The flag is captured once at construction (passed in by Device::initialize from its
    // MetalContext's rtoptions) -- it's a process-wide debug toggle, no need to look it up
    // again per allocation.
    per_pid_tracking_enabled_(!tracking_disabled),
    verbose_enabled_(verbose),
    is_creator_(false) {
    // Format: /tt_device_<chip_unique_id>_memory
    // chip_unique_id from UMD is globally unique and never changes
    std::string shm_name = "/tt_device_" + std::to_string(asic_id) + "_memory";

    // Try exclusive create first to see if we're the first
    shm_fd_ = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (shm_fd_ != -1) {
        is_creator_ = true;
    } else if (errno == EEXIST) {
        // Already exists, just open it
        shm_fd_ = shm_open(shm_name.c_str(), O_RDWR, 0600);
        is_creator_ = false;
    }

    if (shm_fd_ == -1) {
        log_warning(tt::LogMetal, "Failed to create shared memory {}: {}", shm_name, strerror(errno));
        return;
    }

    // Set size (this is idempotent - won't shrink if already larger)
    if (ftruncate(shm_fd_, sizeof(DeviceMemoryRegion)) == -1) {
        log_warning(tt::LogMetal, "Failed to set shared memory size: {}", strerror(errno));
        close(shm_fd_);
        shm_fd_ = -1;
        return;
    }

    // Map into address space
    region_ = static_cast<DeviceMemoryRegion*>(
        mmap(nullptr, sizeof(DeviceMemoryRegion), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0));

    if (region_ == MAP_FAILED) {
        log_warning(tt::LogMetal, "Failed to mmap shared memory: {}", strerror(errno));
        close(shm_fd_);
        shm_fd_ = -1;
        region_ = nullptr;
        return;
    }

    // Decide whether we are the one to initialize the region, or whether we must
    // wait for somebody else to finish doing so.
    //
    // Winning the O_CREAT|O_EXCL race is NOT sufficient to make us the initializer
    // in a race-free way: creation, ftruncate, mmap and initialization are separate
    // steps, so another process can attach and start writing in between. Readiness
    // is therefore published explicitly through init_state, and the right to
    // initialize is claimed with a CAS.
    bool must_initialize = false;
    const uint32_t observed_state = region_->init_state.load(std::memory_order_acquire);

    if (observed_state == SHM_INIT_READY &&
        region_->version.load(std::memory_order_relaxed) == DEVICE_MEMORY_REGION_VERSION) {
        // Fully initialized region of the expected layout: attach to it as-is.
    } else if (observed_state == SHM_INIT_READY) {
        // Ready, but a layout we do not understand. Reclaim it if nobody is using it.
        // reference_count alone cannot decide that -- an older writer leaks it when killed,
        // which would make the upgrade one-way -- so where `processes` is known to be at a
        // known offset, ask whether an owner is still alive instead.
        const uint32_t found_version = region_->version.load(std::memory_order_relaxed);
        const uint32_t attached = region_->reference_count.load(std::memory_order_acquire);
        bool reclaimable = attached == 0;
        if (!reclaimable && process_table_layout_is_known(found_version)) {
            // Read-only: nothing is written to a foreign-layout region unless we go on to
            // reinitialize the whole thing.
            reclaimable = !region_has_live_process();
        }

        uint32_t expected = SHM_INIT_READY;
        if (reclaimable && region_->init_state.compare_exchange_strong(
                               expected, SHM_INIT_IN_PROGRESS, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            log_info(
                tt::LogMetal,
                "SHM version mismatch for asic_id=0x{:x} (found v{}, expected v{}), reinitializing stale region "
                "(reference_count={}, no live owner)",
                asic_id_,
                found_version,
                DEVICE_MEMORY_REGION_VERSION,
                attached);
            must_initialize = true;
        } else {
            log_warning(
                tt::LogMetal,
                "SHM version mismatch for asic_id=0x{:x} (found v{}, expected v{}) and the region is still in use "
                "(reference_count={}); disabling SHM tracking for this provider. If no tt-metal process is running, "
                "the region is stale and can be removed with: rm /dev/shm/tt_device_{}_memory",
                asic_id_,
                found_version,
                DEVICE_MEMORY_REGION_VERSION,
                attached,
                asic_id_);
            munmap(region_, sizeof(DeviceMemoryRegion));
            region_ = nullptr;
            close(shm_fd_);
            shm_fd_ = -1;
            return;
        }
    } else {
        // Zero-filled (brand new) or somebody is mid-initialization.
        uint32_t expected = SHM_INIT_UNINITIALIZED;
        if (region_->init_state.compare_exchange_strong(
                expected, SHM_INIT_IN_PROGRESS, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            must_initialize = true;
        } else if (!wait_for_region_ready()) {
            log_warning(
                tt::LogMetal,
                "Timed out waiting for another process to initialize SHM region for asic_id=0x{:x}; "
                "disabling SHM tracking for this provider",
                asic_id_);
            munmap(region_, sizeof(DeviceMemoryRegion));
            region_ = nullptr;
            close(shm_fd_);
            shm_fd_ = -1;
            return;
        }
    }

    if (must_initialize) {
        initialize_region();
        // Publish last: everything written above must be visible to any process
        // that observes READY.
        region_->init_state.store(SHM_INIT_READY, std::memory_order_release);
    }

    // Reclaim slots belonging to processes that died without cleaning up, then claim
    // our own. Attachment is represented by owning a ProcessStats slot, and
    // reference_count is derived from the live slots, so a crashed process no longer
    // pins the count above zero (which used to stop the region ever resetting).
    if (per_pid_tracking_enabled_) {
        reap_dead_processes();
        claim_own_pid_entry(getpid());
        recompute_aggregates();
    }

    // ALWAYS update identifiers, even when reattaching to existing SHM
    region_->board_serial.store(0, std::memory_order_relaxed);
    region_->asic_id.store(asic_id_, std::memory_order_relaxed);
    TT_ASSERT(device_id_ >= 0, "Negative device_id {} passed to SHM provider", device_id_);
    region_->device_id.store(static_cast<uint32_t>(device_id_), std::memory_order_relaxed);

    if (verbose_enabled_) {
        log_info(
            tt::LogMetal,
            "SHM Provider initialized: device_id={}, asic_id=0x{:x}, shm_name={}, is_creator={}, region_={}, "
            "per_pid_tracking={}",
            device_id_,
            asic_id_,
            shm_name,
            is_creator_,
            (region_ != nullptr && region_ != MAP_FAILED) ? "valid" : "nullptr",
            per_pid_tracking_enabled_);
    }
}

SharedMemoryStatsProvider::~SharedMemoryStatsProvider() {
    if (region_ != nullptr && region_ != MAP_FAILED) {
        const pid_t my_pid = getpid();

        // Release our slot. There is no separate "subtract my totals from the
        // aggregate" step any more: the aggregates are derived from the live slots,
        // so dropping the slot is what removes our contribution. That is also why a
        // SIGKILLed process can be cleaned up by somebody else later -- the old code
        // could only do this subtraction from its own destructor, so a killed process
        // left its allocations in the totals permanently.
        for (auto& slot : region_->processes) {
            if (slot.pid.load(std::memory_order_relaxed) == my_pid) {
                clear_process_slot(slot);
                break;
            }
        }

        // Also drop slots of any process that died in the meantime, so the "no live
        // processes" state below is reached even if a peer was killed.
        reap_dead_processes();
        recompute_aggregates();

        const uint32_t attached = region_->reference_count.load(std::memory_order_relaxed);
        if (attached == 0) {
            log_debug(tt::LogMetal, "Device {}: last process detached, per-chip stats reset", device_id_);
            // Per-chip stats are accumulated rather than per-process attributed, so
            // they cannot be derived. Clearing them when nobody is attached keeps the
            // invariant "no live processes => everything reads zero".
            for (auto& chip_stat : region_->chip_stats) {
                if (chip_stat.chip_id.load(std::memory_order_relaxed) != CHIP_STATS_UNUSED) {
                    chip_stat.dram_allocated.store(0, std::memory_order_relaxed);
                    chip_stat.l1_allocated.store(0, std::memory_order_relaxed);
                    chip_stat.l1_small_allocated.store(0, std::memory_order_relaxed);
                    chip_stat.trace_allocated.store(0, std::memory_order_relaxed);
                    chip_stat.cb_allocated.store(0, std::memory_order_relaxed);
                }
            }
        } else {
            log_debug(tt::LogMetal, "Device {}: process detached, {} still attached", device_id_, attached);
        }

        munmap(region_, sizeof(DeviceMemoryRegion));
        region_ = nullptr;
    }

    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

void SharedMemoryStatsProvider::initialize_region() {
    if (!region_) {
        return;
    }

    // Caller owns init_state: it CASes it to SHM_INIT_IN_PROGRESS before calling us and
    // publishes SHM_INIT_READY afterwards. We must not touch it here, or another process
    // could observe READY while these stores are still in flight.
    //
    // reference_count and num_active_processes are derived from the live process slots
    // (see recompute_aggregates), so zeroing them here is just establishing the
    // "no slots claimed yet" baseline that the loop at the end of this function creates.
    region_->version.store(DEVICE_MEMORY_REGION_VERSION, std::memory_order_relaxed);
    region_->num_active_processes.store(0, std::memory_order_relaxed);
    region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
    region_->reference_count.store(0, std::memory_order_relaxed);

    // Set physical chip identification (for proper device correlation)
    region_->board_serial.store(0, std::memory_order_relaxed);
    region_->asic_id.store(asic_id_, std::memory_order_relaxed);
    TT_ASSERT(device_id_ >= 0, "Negative device_id {} in SHM initialize_region", device_id_);
    region_->device_id.store(static_cast<uint32_t>(device_id_), std::memory_order_relaxed);

    // Initialize atomic counters to zero
    region_->total_dram_allocated.store(0, std::memory_order_relaxed);
    region_->total_l1_allocated.store(0, std::memory_order_relaxed);
    region_->total_l1_small_allocated.store(0, std::memory_order_relaxed);
    region_->total_trace_allocated.store(0, std::memory_order_relaxed);
    region_->total_cb_allocated.store(0, std::memory_order_relaxed);

    // Initialize per-chip entries (for remote device tracking)
    for (auto & chip_stat : region_->chip_stats) {
        chip_stat.chip_id.store(CHIP_STATS_UNUSED, std::memory_order_relaxed);
        chip_stat.is_remote.store(0, std::memory_order_relaxed);
        chip_stat.dram_allocated.store(0, std::memory_order_relaxed);
        chip_stat.l1_allocated.store(0, std::memory_order_relaxed);
        chip_stat.l1_small_allocated.store(0, std::memory_order_relaxed);
        chip_stat.trace_allocated.store(0, std::memory_order_relaxed);
        chip_stat.cb_allocated.store(0, std::memory_order_relaxed);
    }

    // Register the gateway chip itself (chip_id = device_id, is_remote = false)
    region_->chip_stats[0].chip_id.store(static_cast<uint32_t>(device_id_), std::memory_order_relaxed);
    region_->chip_stats[0].is_remote.store(0, std::memory_order_relaxed);

    // Clear per-process entries
    for (auto & processe : region_->processes) {
        processe.pid = 0;  // 0 = unused
        processe.dram_allocated = 0;
        processe.l1_allocated = 0;
        processe.l1_small_allocated = 0;
        processe.trace_allocated = 0;
        processe.cb_allocated = 0;
        processe.last_update_timestamp = 0;
        std::memset(processe.process_name, 0, 64);
    }
}

void SharedMemoryStatsProvider::record_allocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id) {
    if (!region_) {
        if (verbose_enabled_) {
            log_warning(
                tt::LogMetal,
                "SHM record_allocation SKIPPED: region_ is nullptr (pid={}, size={} B, type={}, chip_id={})",
                pid,
                size,
                static_cast<unsigned>(type),
                chip_id);
        }
        return;
    }

    if (verbose_enabled_) {
        static const char* type_names[] = {"DRAM", "L1", "L1_SMALL", "TRACE", "CB"};
        auto type_idx = static_cast<size_t>(type);
        const char* type_name = (type_idx < 5) ? type_names[type_idx] : "UNKNOWN";
        log_info(
            tt::LogMetal,
            "SHM record_allocation: pid={}, type={}, size={} B ({} KB), chip_id={}, device_id={}, asic_id=0x{:x}",
            pid,
            type_name,
            size,
            size / 1024,
            chip_id,
            device_id_,
            asic_id_);
    }

    // Update aggregated counters (always - this is the fast path)
    switch (type) {
        case ShmBufferType::DRAM: region_->total_dram_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::L1: region_->total_l1_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::L1_SMALL:
            region_->total_l1_small_allocated.fetch_add(size, std::memory_order_relaxed);
            break;
        case ShmBufferType::TRACE: region_->total_trace_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::CB: region_->total_cb_allocated.fetch_add(size, std::memory_order_relaxed); break;
        default: break;
    }

    // Update per-chip counters (for remote device tracking)
    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        switch (type) {
            case ShmBufferType::DRAM: chip_entry->dram_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::L1: chip_entry->l1_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::L1_SMALL:
                chip_entry->l1_small_allocated.fetch_add(size, std::memory_order_relaxed);
                break;
            case ShmBufferType::TRACE: chip_entry->trace_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::CB: chip_entry->cb_allocated.fetch_add(size, std::memory_order_relaxed); break;
            default: break;
        }
    }

    // Update timestamp
    region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);

    // Update per-PID stats if enabled
    if (per_pid_tracking_enabled_) {
        auto* pid_entry = find_or_create_pid_entry(pid);
        if (pid_entry) {
            switch (type) {
                case ShmBufferType::DRAM: pid_entry->dram_allocated.fetch_add(size, std::memory_order_relaxed); break;
                case ShmBufferType::L1: pid_entry->l1_allocated.fetch_add(size, std::memory_order_relaxed); break;
                case ShmBufferType::L1_SMALL:
                    pid_entry->l1_small_allocated.fetch_add(size, std::memory_order_relaxed);
                    break;
                case ShmBufferType::TRACE: pid_entry->trace_allocated.fetch_add(size, std::memory_order_relaxed); break;
                case ShmBufferType::CB: pid_entry->cb_allocated.fetch_add(size, std::memory_order_relaxed); break;
                default: break;
            }
            pid_entry->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
        }
    }
}

void SharedMemoryStatsProvider::record_deallocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id) {
    if (!region_) {
        if (verbose_enabled_) {
            log_warning(
                tt::LogMetal,
                "SHM record_deallocation SKIPPED: region_ is nullptr (pid={}, size={} B, type={}, chip_id={})",
                pid,
                size,
                static_cast<unsigned>(type),
                chip_id);
        }
        return;
    }

    if (verbose_enabled_) {
        static const char* type_names[] = {"DRAM", "L1", "L1_SMALL", "TRACE", "CB"};
        auto type_idx = static_cast<size_t>(type);
        const char* type_name = (type_idx < 5) ? type_names[type_idx] : "UNKNOWN";
        log_info(
            tt::LogMetal,
            "SHM record_deallocation: pid={}, type={}, size={} B ({} KB), chip_id={}, device_id={}, asic_id=0x{:x}",
            pid,
            type_name,
            size,
            size / 1024,
            chip_id,
            device_id_,
            asic_id_);
    }

    // Update aggregated counters with underflow protection
    // Note: We use compare-and-swap loop to prevent underflow
    auto safe_sub = [](std::atomic<uint64_t>& counter, uint64_t size) {
        uint64_t current = counter.load(std::memory_order_relaxed);
        uint64_t new_val;
        do {
            if (current < size) {
                // Underflow would occur - clamp to 0
                new_val = 0;
            } else {
                new_val = current - size;
            }
        } while (
            !counter.compare_exchange_weak(current, new_val, std::memory_order_relaxed, std::memory_order_relaxed));
    };

    switch (type) {
        case ShmBufferType::DRAM: safe_sub(region_->total_dram_allocated, size); break;
        case ShmBufferType::L1: safe_sub(region_->total_l1_allocated, size); break;
        case ShmBufferType::L1_SMALL: safe_sub(region_->total_l1_small_allocated, size); break;
        case ShmBufferType::TRACE: safe_sub(region_->total_trace_allocated, size); break;
        case ShmBufferType::CB: safe_sub(region_->total_cb_allocated, size); break;
        default: break;
    }

    // Update per-chip counters (with underflow protection)
    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        switch (type) {
            case ShmBufferType::DRAM: safe_sub(chip_entry->dram_allocated, size); break;
            case ShmBufferType::L1: safe_sub(chip_entry->l1_allocated, size); break;
            case ShmBufferType::L1_SMALL: safe_sub(chip_entry->l1_small_allocated, size); break;
            case ShmBufferType::TRACE: safe_sub(chip_entry->trace_allocated, size); break;
            case ShmBufferType::CB: safe_sub(chip_entry->cb_allocated, size); break;
            default: break;
        }
    }

    // Update timestamp
    region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);

    // Update per-PID stats if enabled (with underflow protection using atomics)
    if (per_pid_tracking_enabled_) {
        auto* pid_entry = find_or_create_pid_entry(pid);
        if (pid_entry) {
            // Helper lambda for atomic subtraction with underflow protection
            auto safe_atomic_sub = [](std::atomic<uint64_t>& counter, uint64_t size) {
                uint64_t current = counter.load(std::memory_order_relaxed);
                uint64_t new_val;
                do {
                    new_val = (current >= size) ? (current - size) : 0;
                } while (!counter.compare_exchange_weak(current, new_val, std::memory_order_relaxed));
            };

            switch (type) {
                case ShmBufferType::DRAM: safe_atomic_sub(pid_entry->dram_allocated, size); break;
                case ShmBufferType::L1: safe_atomic_sub(pid_entry->l1_allocated, size); break;
                case ShmBufferType::L1_SMALL: safe_atomic_sub(pid_entry->l1_small_allocated, size); break;
                case ShmBufferType::TRACE: safe_atomic_sub(pid_entry->trace_allocated, size); break;
                case ShmBufferType::CB: safe_atomic_sub(pid_entry->cb_allocated, size); break;
                default: break;
            }
            pid_entry->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
        }
    }
}

SharedMemoryStatsProvider::DeviceStats SharedMemoryStatsProvider::get_device_stats() const {
    if (!region_) {
        return {0, 0, 0, 0, 0, 0};
    }

    return {
        region_->total_dram_allocated.load(std::memory_order_relaxed),
        region_->total_l1_allocated.load(std::memory_order_relaxed),
        region_->total_l1_small_allocated.load(std::memory_order_relaxed),
        region_->total_trace_allocated.load(std::memory_order_relaxed),
        region_->total_cb_allocated.load(std::memory_order_relaxed),
        region_->last_update_timestamp.load(std::memory_order_relaxed)};
}

std::vector<SharedMemoryStatsProvider::ProcessInfo> SharedMemoryStatsProvider::get_process_stats() const {
    std::vector<ProcessInfo> result;

    if (!region_) {
        return result;
    }

    result.reserve(MAX_PROCESSES);
    for (auto & processe : region_->processes) {
        if (processe.pid.load(std::memory_order_relaxed) != 0) {
            ProcessInfo info;
            info.pid = processe.pid.load(std::memory_order_relaxed);
            info.dram_allocated = processe.dram_allocated.load(std::memory_order_relaxed);
            info.l1_allocated = processe.l1_allocated.load(std::memory_order_relaxed);
            info.l1_small_allocated = processe.l1_small_allocated.load(std::memory_order_relaxed);
            info.trace_allocated = processe.trace_allocated.load(std::memory_order_relaxed);
            info.cb_allocated = processe.cb_allocated.load(std::memory_order_relaxed);
            info.timestamp = processe.last_update_timestamp.load(std::memory_order_relaxed);
            info.process_name = std::string(processe.process_name);
            result.push_back(info);
        }
    }

    return result;
}

DeviceMemoryRegion::ProcessStats* SharedMemoryStatsProvider::find_or_create_pid_entry(pid_t pid) {
    if (!region_) {
        return nullptr;
    }

    // First, try to find existing entry
    for (auto & processe : region_->processes) {
        if (processe.pid == pid) {
            return &processe;
        }
    }

    // Not found. Normally our slot was claimed at attach time, so getting here means
    // the slot table was full then; try again (a peer may have exited since).
    return claim_own_pid_entry(pid);
}

DeviceMemoryRegion::ProcessStats* SharedMemoryStatsProvider::claim_own_pid_entry(pid_t pid) {
    if (!region_) {
        return nullptr;
    }

    // Already ours?
    for (auto& slot : region_->processes) {
        if (slot.pid.load(std::memory_order_relaxed) == pid) {
            return &slot;
        }
    }

    // Claim a free slot. CAS so that processes racing for the same slot cannot both win.
    for (auto& slot : region_->processes) {
        pid_t expected = 0;
        if (slot.pid.compare_exchange_strong(expected, pid, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            slot.dram_allocated.store(0, std::memory_order_relaxed);
            slot.l1_allocated.store(0, std::memory_order_relaxed);
            slot.l1_small_allocated.store(0, std::memory_order_relaxed);
            slot.trace_allocated.store(0, std::memory_order_relaxed);
            slot.cb_allocated.store(0, std::memory_order_relaxed);
            slot.last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);

            const std::string proc_name = get_process_name(pid);
            strncpy(slot.process_name, proc_name.c_str(), 63);
            slot.process_name[63] = '\0';
            return &slot;
        }
    }

    // Slot table full even after reaping. Track nothing for this process rather than
    // corrupting somebody else's slot; aggregates will simply not include us.
    log_warning(
        tt::LogMetal,
        "SHM process table full ({} slots) for asic_id=0x{:x}; memory stats for pid {} will not be reported",
        MAX_PROCESSES,
        asic_id_,
        pid);
    return nullptr;
}

void SharedMemoryStatsProvider::clear_process_slot(DeviceMemoryRegion::ProcessStats& slot) {
    slot.dram_allocated.store(0, std::memory_order_relaxed);
    slot.l1_allocated.store(0, std::memory_order_relaxed);
    slot.l1_small_allocated.store(0, std::memory_order_relaxed);
    slot.trace_allocated.store(0, std::memory_order_relaxed);
    slot.cb_allocated.store(0, std::memory_order_relaxed);
    slot.last_update_timestamp.store(0, std::memory_order_relaxed);
    std::memset(slot.process_name, 0, sizeof(slot.process_name));
    // Release the slot last: pid == 0 is what makes it claimable, so it must not
    // become visible before the fields above have been cleared.
    slot.pid.store(0, std::memory_order_release);
}

bool SharedMemoryStatsProvider::process_table_layout_is_known(uint32_t version) {
    // v2..v4 place `processes` at the same offset with the same ProcessStats layout. v1
    // predates it, and a higher version belongs to a writer newer than us, so in both cases we
    // must not assume where the table is.
    return version >= 2 && version <= DEVICE_MEMORY_REGION_VERSION;
}

bool SharedMemoryStatsProvider::region_has_live_process() const {
    if (!region_) {
        return false;
    }
    for (const auto& slot : region_->processes) {
        const pid_t pid = slot.pid.load(std::memory_order_acquire);
        if (pid != 0 && pid != getpid() && process_is_alive(pid)) {
            return true;
        }
    }
    return false;
}

bool SharedMemoryStatsProvider::process_is_alive(pid_t pid) {
    if (pid <= 0) {
        return false;
    }
    // Signal 0 performs the permission/existence check without delivering anything.
    // EPERM means the PID exists but belongs to another user, which still counts as alive.
    return ::kill(pid, 0) == 0 || errno == EPERM;
}

size_t SharedMemoryStatsProvider::reap_dead_processes() {
    if (!region_) {
        return 0;
    }

    size_t reaped = 0;
    for (auto& slot : region_->processes) {
        const pid_t pid = slot.pid.load(std::memory_order_acquire);
        if (pid == 0 || pid == getpid() || process_is_alive(pid)) {
            continue;
        }
        log_debug(
            tt::LogMetal,
            "Reclaiming SHM slot of dead pid {} for asic_id=0x{:x} (process exited without cleanup)",
            pid,
            asic_id_);
        clear_process_slot(slot);
        reaped++;
    }
    return reaped;
}

void SharedMemoryStatsProvider::recompute_aggregates() {
    // Deriving the totals requires the per-process slots to be the source of truth.
    // With per-PID tracking off nothing populates them, so recomputing would zero
    // out figures that record_allocation() is maintaining incrementally instead.
    if (!region_ || !per_pid_tracking_enabled_) {
        return;
    }

    uint64_t dram = 0;
    uint64_t l1 = 0;
    uint64_t l1_small = 0;
    uint64_t trace = 0;
    uint64_t cb = 0;
    uint32_t live = 0;

    for (const auto& slot : region_->processes) {
        if (slot.pid.load(std::memory_order_acquire) == 0) {
            continue;
        }
        live++;
        dram += slot.dram_allocated.load(std::memory_order_relaxed);
        l1 += slot.l1_allocated.load(std::memory_order_relaxed);
        l1_small += slot.l1_small_allocated.load(std::memory_order_relaxed);
        trace += slot.trace_allocated.load(std::memory_order_relaxed);
        cb += slot.cb_allocated.load(std::memory_order_relaxed);
    }

    region_->total_dram_allocated.store(dram, std::memory_order_relaxed);
    region_->total_l1_allocated.store(l1, std::memory_order_relaxed);
    region_->total_l1_small_allocated.store(l1_small, std::memory_order_relaxed);
    region_->total_trace_allocated.store(trace, std::memory_order_relaxed);
    region_->total_cb_allocated.store(cb, std::memory_order_relaxed);
    region_->num_active_processes.store(live, std::memory_order_relaxed);
    region_->reference_count.store(live, std::memory_order_release);
    region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
}

bool SharedMemoryStatsProvider::wait_for_region_ready() {
    if (!region_) {
        return false;
    }
    // Initialization is a handful of stores; a short bounded wait is plenty, and
    // bounding it means a process that died mid-initialization cannot hang us.
    constexpr int kMaxAttempts = 1000;  // 1000 x 1ms = 1s
    for (int i = 0; i < kMaxAttempts; i++) {
        if (region_->init_state.load(std::memory_order_acquire) == SHM_INIT_READY) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

uint64_t SharedMemoryStatsProvider::current_timestamp_ns() {
    auto now = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(ns);
}

std::string SharedMemoryStatsProvider::get_process_name(pid_t pid) {
    std::string path = "/proc/" + std::to_string(pid) + "/comm";
    std::ifstream file(path);
    if (!file) {
        return "unknown";
    }

    std::string name;
    std::getline(file, name);
    return name;
}

DeviceMemoryRegion::ChipStats* SharedMemoryStatsProvider::find_or_create_chip_entry(uint32_t chip_id) {
    if (!region_) {
        return nullptr;
    }

    // First, try to find existing entry
    for (auto & chip_stat : region_->chip_stats) {
        if (chip_stat.chip_id.load(std::memory_order_relaxed) == chip_id) {
            return &chip_stat;
        }
    }

    // Not found — claim a slot with CAS to avoid TOCTOU race between threads
    for (auto & chip_stat : region_->chip_stats) {
        uint32_t expected = CHIP_STATS_UNUSED;
        if (chip_stat.chip_id.compare_exchange_strong(
                expected, chip_id, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            // We claimed this slot; initialize it
            chip_stat.is_remote.store(0, std::memory_order_relaxed);
            chip_stat.dram_allocated.store(0, std::memory_order_relaxed);
            chip_stat.l1_allocated.store(0, std::memory_order_relaxed);
            chip_stat.l1_small_allocated.store(0, std::memory_order_relaxed);
            chip_stat.trace_allocated.store(0, std::memory_order_relaxed);
            chip_stat.cb_allocated.store(0, std::memory_order_relaxed);
            return &chip_stat;
        }
        if (expected == chip_id) {
            // Another thread claimed this slot for the same chip_id concurrently
            return &chip_stat;
        }
    }

    // No free slots
    return nullptr;
}

void SharedMemoryStatsProvider::register_chip(uint32_t chip_id, bool is_remote) {
    if (!region_) {
        return;
    }

    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        chip_entry->is_remote.store(is_remote ? 1u : 0u, std::memory_order_relaxed);
    }
}

std::vector<SharedMemoryStatsProvider::ChipInfo> SharedMemoryStatsProvider::get_chip_stats() const {
    std::vector<ChipInfo> result;
    if (!region_) {
        return result;
    }

    result.reserve(MAX_CHIPS_PER_DEVICE);
    for (auto & chip_stat : region_->chip_stats) {
        if (chip_stat.chip_id.load(std::memory_order_relaxed) != CHIP_STATS_UNUSED) {
            ChipInfo info{};
            info.chip_id = chip_stat.chip_id.load(std::memory_order_relaxed);
            info.is_remote = (chip_stat.is_remote.load(std::memory_order_relaxed) != 0);
            info.dram_allocated = chip_stat.dram_allocated.load(std::memory_order_relaxed);
            info.l1_allocated = chip_stat.l1_allocated.load(std::memory_order_relaxed);
            info.l1_small_allocated = chip_stat.l1_small_allocated.load(std::memory_order_relaxed);
            info.trace_allocated = chip_stat.trace_allocated.load(std::memory_order_relaxed);
            info.cb_allocated = chip_stat.cb_allocated.load(std::memory_order_relaxed);
            result.push_back(info);
        }
    }

    return result;
}

}  // namespace tt::tt_metal
