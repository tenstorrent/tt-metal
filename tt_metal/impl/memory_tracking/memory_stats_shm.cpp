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
#include <csignal>
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

// Subtract without wrapping. The counters are unsigned and shared across processes, so a
// subtraction that would go negative -- possible if a process is killed between recording an
// allocation and updating its own slot -- must clamp rather than wrap to ~1.8e19.
void SharedMemoryStatsProvider::saturating_sub(std::atomic<uint64_t>& counter, uint64_t amount) {
    uint64_t current = counter.load(std::memory_order_relaxed);
    uint64_t updated = 0;
    do {
        updated = (current < amount) ? 0 : current - amount;
    } while (!counter.compare_exchange_weak(current, updated, std::memory_order_relaxed, std::memory_order_relaxed));
}

void SharedMemoryStatsProvider::saturating_sub(std::atomic<uint32_t>& counter, uint32_t amount) {
    uint32_t current = counter.load(std::memory_order_relaxed);
    uint32_t updated = 0;
    do {
        updated = (current < amount) ? 0 : current - amount;
    } while (!counter.compare_exchange_weak(current, updated, std::memory_order_relaxed, std::memory_order_relaxed));
}

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
        // Ready, but a layout we do not understand. Reclaim it only if nobody is using it;
        // see legacy_region_is_reclaimable() for what decides that. The test is read-only:
        // nothing is written to a foreign-layout region unless we go on to reinitialize the
        // whole thing.
        const uint32_t found_version = region_->version.load(std::memory_order_relaxed);
        const uint32_t attached = region_->reference_count.load(std::memory_order_acquire);
        const bool reclaimable = legacy_region_is_reclaimable(found_version);

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
            detach_region();
            return;
        }
    } else {
        // Zero-filled (brand new), somebody is mid-initialization, or -- the case this
        // branch has to be careful about -- a region written by a layout that predates
        // init_state.
        //
        // A pre-v4 region publishes no init_state at all: the field is appended last, so it
        // reads UNINITIALIZED whether it lands in that layout's tail padding or in bytes
        // ftruncate appended here. Either way those bytes were zero-filled at creation and
        // never written, so init_state alone cannot tell a fresh region from a live v2/v3
        // one -- and claiming initialization on the strength of it would run
        // initialize_region() over a region an older process is still writing, zeroing its
        // totals and its whole process table.
        //
        // `version` is the field that does separate them: an initializer publishes it before
        // anything observes the region as ready, so nonzero means some layout wrote here.
        // Such a region may only be taken over on the live-owner test, exactly like the
        // wrong-version READY case above.
        const uint32_t found_version = region_->version.load(std::memory_order_acquire);
        if (found_version != 0 && found_version != DEVICE_MEMORY_REGION_VERSION &&
            !legacy_region_is_reclaimable(found_version)) {
            log_warning(
                tt::LogMetal,
                "SHM region for asic_id=0x{:x} was written by an older layout (found v{}, expected v{}) that "
                "publishes no readiness flag, and a live process still owns a slot in it; disabling SHM tracking "
                "for this provider rather than reinitializing it underneath that process. If no tt-metal process "
                "is running, the region is stale and can be removed with: rm /dev/shm/tt_device_{}_memory",
                asic_id_,
                found_version,
                DEVICE_MEMORY_REGION_VERSION,
                asic_id_);
            detach_region();
            return;
        }

        uint32_t expected = SHM_INIT_UNINITIALIZED;
        if (region_->init_state.compare_exchange_strong(
                expected, SHM_INIT_IN_PROGRESS, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            must_initialize = true;
        } else if (!wait_for_region_ready()) {
            // Nobody published READY. Either the initializer is pathologically slow, or it died
            // between claiming initialization and publishing -- which leaves init_state stuck at
            // IN_PROGRESS and would wedge this region for every later process, exactly the
            // one-way dead end that a leaked reference_count used to cause.
            //
            // Take it over when nothing can be using it -- the same test the branches above
            // use. Reset to UNINITIALIZED and re-claim, so that two processes timing out
            // together still leave exactly one initializer: the loser's CAS fails and it waits
            // for READY.
            bool taken_over = false;
            const uint32_t stalled_version = region_->version.load(std::memory_order_relaxed);
            if (region_->init_state.load(std::memory_order_acquire) == SHM_INIT_IN_PROGRESS &&
                legacy_region_is_reclaimable(stalled_version)) {
                uint32_t expected = SHM_INIT_IN_PROGRESS;
                if (region_->init_state.compare_exchange_strong(
                        expected, SHM_INIT_UNINITIALIZED, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    expected = SHM_INIT_UNINITIALIZED;
                    taken_over = region_->init_state.compare_exchange_strong(
                        expected, SHM_INIT_IN_PROGRESS, std::memory_order_acq_rel, std::memory_order_relaxed);
                }
            }

            if (taken_over) {
                log_warning(
                    tt::LogMetal,
                    "SHM region for asic_id=0x{:x} was left mid-initialization by a process that did not "
                    "finish; reinitializing it",
                    asic_id_);
                must_initialize = true;
            } else {
                log_warning(
                    tt::LogMetal,
                    "Timed out waiting for another process to initialize SHM region for asic_id=0x{:x}; "
                    "disabling SHM tracking for this provider",
                    asic_id_);
                detach_region();
                return;
            }
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
        // Reaping may have emptied the region. Do the idle reset before claiming our own
        // slot, so that whatever a killed predecessor left in the aggregates is cleared
        // rather than carried into this run's figures.
        reset_aggregates_if_idle();
        claim_own_pid_entry(getpid());
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

        // Release our slot, which subtracts our bytes from the device-wide totals. The
        // same subtraction is what reap_dead_processes() applies on behalf of a process
        // that never got here, so a SIGKILLed run can be cleaned up by whoever attaches
        // next instead of leaving its allocations in the totals permanently.
        for (auto& slot : region_->processes) {
            if (slot.pid.load(std::memory_order_relaxed) == my_pid) {
                release_process_slot(slot, my_pid);
                break;
            }
        }

        // Also drop slots of any process that died in the meantime, so the "no live
        // processes" state below is reached even if a peer was killed.
        reap_dead_processes();

        reset_aggregates_if_idle();

        munmap(region_, sizeof(DeviceMemoryRegion));
        region_ = nullptr;
    }

    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

void SharedMemoryStatsProvider::reset_aggregates_if_idle() {
    if (!region_) {
        return;
    }
    const uint32_t attached = region_->reference_count.load(std::memory_order_acquire);
    if (attached != 0) {
        log_debug(tt::LogMetal, "Device {}: process detached, {} still attached", device_id_, attached);
        return;
    }

    log_debug(tt::LogMetal, "Device {}: no process attached, aggregates reset", device_id_);

    // A process can claim a slot while these stores are in flight. That costs nothing: a
    // freshly claimed slot has all its byte counters zeroed by claim_own_pid_entry() before
    // it is published, so there is no contribution here yet to erase.
    region_->total_dram_allocated.store(0, std::memory_order_relaxed);
    region_->total_l1_allocated.store(0, std::memory_order_relaxed);
    region_->total_l1_small_allocated.store(0, std::memory_order_relaxed);
    region_->total_trace_allocated.store(0, std::memory_order_relaxed);
    region_->total_cb_allocated.store(0, std::memory_order_relaxed);

    for (auto& chip_stat : region_->chip_stats) {
        if (chip_stat.chip_id.load(std::memory_order_relaxed) != CHIP_STATS_UNUSED) {
            chip_stat.dram_allocated.store(0, std::memory_order_relaxed);
            chip_stat.l1_allocated.store(0, std::memory_order_relaxed);
            chip_stat.l1_small_allocated.store(0, std::memory_order_relaxed);
            chip_stat.trace_allocated.store(0, std::memory_order_relaxed);
            chip_stat.cb_allocated.store(0, std::memory_order_relaxed);
        }
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
    // reference_count and num_active_processes are maintained by delta as processes claim
    // and release slots, so zeroing them here just establishes the "no slots claimed yet"
    // baseline that the loop at the end of this function creates.
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

    // Find our slot FIRST. Every aggregate here moves by delta, and the only thing that ever
    // subtracts these bytes again is the release of this slot -- so recording into the totals
    // without one inflates them for the life of the region. With the table full (64 processes,
    // or dead slots not yet reaped) it is better to report nothing for this process than to
    // corrupt the device-wide figure everybody else reads.
    DeviceMemoryRegion::ProcessStats* pid_entry = nullptr;
    if (per_pid_tracking_enabled_) {
        pid_entry = find_or_create_pid_entry(pid);
        if (pid_entry == nullptr) {
            return;  // claim_own_pid_entry has already warned about the full table
        }
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
    if (pid_entry != nullptr) {
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

    // Symmetric with record_allocation: a process that never got a slot never added these
    // bytes to the totals, so it must not subtract them either.
    DeviceMemoryRegion::ProcessStats* pid_entry = nullptr;
    if (per_pid_tracking_enabled_) {
        pid_entry = find_or_create_pid_entry(pid);
        if (pid_entry == nullptr) {
            return;
        }
    }

    // Aggregated counters, clamped so a stale or duplicated free cannot wrap them.
    switch (type) {
        case ShmBufferType::DRAM: saturating_sub(region_->total_dram_allocated, size); break;
        case ShmBufferType::L1: saturating_sub(region_->total_l1_allocated, size); break;
        case ShmBufferType::L1_SMALL: saturating_sub(region_->total_l1_small_allocated, size); break;
        case ShmBufferType::TRACE: saturating_sub(region_->total_trace_allocated, size); break;
        case ShmBufferType::CB: saturating_sub(region_->total_cb_allocated, size); break;
        default: break;
    }

    // Update per-chip counters (with underflow protection)
    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        switch (type) {
            case ShmBufferType::DRAM: saturating_sub(chip_entry->dram_allocated, size); break;
            case ShmBufferType::L1: saturating_sub(chip_entry->l1_allocated, size); break;
            case ShmBufferType::L1_SMALL: saturating_sub(chip_entry->l1_small_allocated, size); break;
            case ShmBufferType::TRACE: saturating_sub(chip_entry->trace_allocated, size); break;
            case ShmBufferType::CB: saturating_sub(chip_entry->cb_allocated, size); break;
            default: break;
        }
    }

    // Update timestamp
    region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);

    // Update per-PID stats if enabled (with underflow protection using atomics)
    if (pid_entry != nullptr) {
        {
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
        if (processe.pid.load(std::memory_order_relaxed) > 0) {
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

            // One more process attached. Only on a fresh claim: re-attaching a slot this
            // process already owns must not double-count it.
            region_->reference_count.fetch_add(1, std::memory_order_release);
            region_->num_active_processes.fetch_add(1, std::memory_order_relaxed);
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

void SharedMemoryStatsProvider::release_process_slot(
    DeviceMemoryRegion::ProcessStats& slot, pid_t expected_owner) {
    if (!region_) {
        return;
    }

    // Claim the release before touching anything. Two attachers can decide to reap the same
    // dead slot at the same moment; without this both would subtract from reference_count and
    // num_active_processes -- the byte totals are safe either way, since only one exchange
    // returns the value -- and the counts would then read lower than the number of attached
    // processes. An under-counted reference_count is not cosmetic: the version-mismatch path
    // treats zero as "nobody is using this" and reinitializes.
    //
    // CAS to SHM_SLOT_RELEASING rather than straight to 0, so the slot cannot be claimed by a
    // new process while its counters are still being cleared. The loser of the CAS returns
    // without doing anything, which also stops it from clearing a slot that the winner has
    // already handed to somebody else.
    // CAS from the pid the caller decided about, NOT from whatever the slot holds now. Reading
    // it here instead let a second reaper of one dead slot clear the live process that had
    // claimed the slot in the meantime: two reapers both see dead pid D, the first releases it,
    // a new process claims the freed slot, and the second then clears that new process's slot
    // and decremented the counts for it. The victim stayed attached and counted while owning
    // nothing, which is the reference_count-above-slots mismatch this used to produce.
    if (expected_owner <= 0) {
        return;
    }
    pid_t owner = expected_owner;
    if (!slot.pid.compare_exchange_strong(
            owner, SHM_SLOT_RELEASING, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;  // no longer that process's slot: released and re-claimed since the decision
    }

    // Subtract this slot's contribution from the device-wide totals before dropping it.
    //
    // Every mutation of an aggregate is a delta -- fetch_add on allocation, saturating
    // subtract on deallocation and here. Deriving the totals instead (sum the live slots,
    // then store) would race every concurrent record_allocation: an allocation landing
    // between another writer's sum and its store is silently dropped from the total and
    // stays dropped. Deltas commute, so there is no such window.
    // Exchange rather than load-then-store: the value taken out of the slot is exactly the
    // value subtracted from the total. Reading the slot, subtracting, and clearing it as three
    // steps would leak the delta of any allocation recorded in between -- and this process can
    // still be allocating on another thread while a device is being closed.
    saturating_sub(region_->total_dram_allocated, slot.dram_allocated.exchange(0, std::memory_order_relaxed));
    saturating_sub(region_->total_l1_allocated, slot.l1_allocated.exchange(0, std::memory_order_relaxed));
    saturating_sub(region_->total_l1_small_allocated, slot.l1_small_allocated.exchange(0, std::memory_order_relaxed));
    saturating_sub(region_->total_trace_allocated, slot.trace_allocated.exchange(0, std::memory_order_relaxed));
    saturating_sub(region_->total_cb_allocated, slot.cb_allocated.exchange(0, std::memory_order_relaxed));

    slot.last_update_timestamp.store(0, std::memory_order_relaxed);
    std::memset(slot.process_name, 0, sizeof(slot.process_name));
    // Release the slot last: pid == 0 is what makes it claimable, so it must not
    // become visible before the fields above have been cleared.
    slot.pid.store(0, std::memory_order_release);

    // One fewer process attached. Counted by delta for the same reason as the byte totals.
    saturating_sub(region_->reference_count, 1);
    saturating_sub(region_->num_active_processes, 1);
    region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
}

bool SharedMemoryStatsProvider::process_table_layout_is_known(uint32_t version) {
    // v2..v4 place `processes` at the same offset with the same ProcessStats layout. v1
    // predates it, and a higher version belongs to a writer newer than us, so in both cases we
    // must not assume where the table is.
    return version >= 2 && version <= DEVICE_MEMORY_REGION_VERSION;
}

void SharedMemoryStatsProvider::detach_region() {
    if (region_ != nullptr && region_ != MAP_FAILED) {
        munmap(region_, sizeof(DeviceMemoryRegion));
    }
    region_ = nullptr;
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

bool SharedMemoryStatsProvider::legacy_region_is_reclaimable(uint32_t found_version) const {
    if (!region_) {
        return false;
    }
    if (found_version == 0) {
        // Never initialized by anybody: the region is zero-filled and there is nothing to lose.
        return true;
    }
    if (process_table_layout_is_known(found_version)) {
        // The table is the authority. reference_count is not: an older writer leaks it
        // permanently when killed, which would make the upgrade one-way.
        return !region_has_live_process();
    }
    // A layout that puts `processes` somewhere we cannot guess (v1, or a writer newer than
    // us). All that is left is the count, and only its zero is safe to act on.
    return region_->reference_count.load(std::memory_order_acquire) == 0;
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
        if (pid <= 0 || pid == getpid() || process_is_alive(pid)) {
            continue;  // free, mid-release by somebody else, ours, or still running
        }
        log_debug(
            tt::LogMetal,
            "Reclaiming SHM slot of dead pid {} for asic_id=0x{:x} (process exited without cleanup)",
            pid,
            asic_id_);
        release_process_slot(slot, pid);
        reaped++;
    }
    return reaped;
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
