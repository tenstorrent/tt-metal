// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/device/device_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include <tt-logger/tt-logger.hpp>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace tt::tt_metal {

// Implementation of SharedMemoryStatsProvider::update_from_allocator()
// Query ONLY locally-allocated CBs (device->get_total_cb_allocated() only counts local CBs)
// Globally-allocated CBs create L1 Buffers and are already tracked in L1 column
void SharedMemoryStatsProvider::update_from_allocator(const Device* device, pid_t pid) {
    if (!region_ || !device) {
        return;
    }

    // Deliberately unthrottled: this is now an atomic load plus a scan of the 64 process
    // slots, and a dropped update is never retried -- a workload that goes idle would leave a
    // stale figure in shared memory forever.
    try {
        // Query actual LOCALLY-allocated CB usage (globally-allocated CBs are in L1 already)
        uint64_t cb_allocated = device->get_total_cb_allocated();

        // Publish our own figure, and move the device-wide total by the same difference.
        //
        // This value is per-process by construction: get_total_cb_allocated() only sees the
        // programs this process dispatched. It is an absolute figure rather than a delta, so
        // the slot is exchanged and the *difference* applied to the device-wide total. That
        // keeps every aggregate mutation a delta, which is what lets concurrent writers
        // coexist: summing the slots and storing the result would drop any allocation another
        // process recorded between the sum and the store.
        //
        // It also fixes the original bug here, which store()d our own value straight into
        // total_cb_allocated -- making the device-wide figure last-writer-wins rather than a
        // total across processes.
        //
        // The exchange and the aggregate move are two separate atomics, and nothing makes the
        // pair atomic against a process dying between them: killed there, the slot holds the
        // new value while the total still reflects the old one, and the reaper subtracts the
        // slot -- so the total ends up off by the delta. record_allocation() has the mirror of
        // this. Closing it needs either a region-wide lock on the record path, which is the
        // cost this rework removed, or a redo log, which cannot be made atomic at every step
        // either. What is done instead is to bound how long a wrong figure survives:
        // saturating_sub keeps it from wrapping, and reset_aggregates_if_idle() zeroes every
        // total once the last process detaches, so drift cannot outlive the busy period that
        // produced it.
        for (auto& slot : region_->processes) {
            if (slot.pid.load(std::memory_order_relaxed) == pid) {
                const uint64_t previous = slot.cb_allocated.exchange(cb_allocated, std::memory_order_relaxed);
                if (cb_allocated >= previous) {
                    region_->total_cb_allocated.fetch_add(cb_allocated - previous, std::memory_order_relaxed);
                } else {
                    saturating_sub(region_->total_cb_allocated, previous - cb_allocated);
                }
                slot.last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
                break;
            }
        }
        region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);

        // Per-chip CB stats for this device. Use the CAS-based claim helper rather than
        // grabbing the first matching-or-unused slot with a plain store, which raced
        // find_or_create_chip_entry() and could let two processes claim the same slot.
        const uint32_t chip_id = static_cast<uint32_t>(device->id());
        if (auto* chip_entry = find_or_create_chip_entry(chip_id)) {
            // Local (gateway) chip: its CB total is the device-wide total we just derived.
            chip_entry->cb_allocated.store(
                region_->total_cb_allocated.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    } catch (const std::exception& e) {
        log_warning(LogMetal, "Failed to query locally-allocated CB stats: {}", e.what());
    }
}

}  // namespace tt::tt_metal
