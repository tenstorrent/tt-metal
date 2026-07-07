// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Emule sanitizer logic (see emule_sanitizers.hpp). Every entry point is a
// no-op when ASAN is disabled, so launch_cores can call them unconditionally
// and stay free of `if (asan_enabled)` clutter. Per-check detail lives in
// SANITIZER_CHECKS.md.

#include "emule_sanitizers.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>

#include "host_sanitizers.hpp"  // emule_asan_enabled / dirty_cb_check_skipped
#include "impl/emulation/emule_live_ranges.hpp"
#include "tt_emule/cb_sync_state.hpp"

// Defined in emule_asan_panic.cpp (same libtt_metal).
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...);

namespace tt::tt_metal::emule {

void ObjectIntentTracker::pre_launch_snapshot(
    const EmuleOobTensorState& oob,
    std::size_t num_kernels,
    const std::vector<uint32_t>& single_kernel_rt_args,
    const uint8_t* l1_data,
    const std::vector<uint64_t>& persistent_cb_ranges,
    [[maybe_unused]] uint32_t lx,
    [[maybe_unused]] uint32_t ly) {
    if (!oob.object_intent_strict || oob.tensor_ranges == nullptr) {
        return;
    }
    if (num_kernels != 1) {
        // Byte changes can't be attributed to one kernel when several share a core,
        // so skip without snapshotting (verify_post_launch then no-ops). See §12.
        return;
    }
    // Exempt I/O tensors handed to this kernel: a live-tensor start that appears in
    // the runtime args is a buffer the kernel was told to operate on. See §12.
    std::unordered_set<uint32_t> io_arg_starts(single_kernel_rt_args.begin(), single_kernel_rt_args.end());
    snapshots_.reserve(oob.tensor_ranges_count);
    for (uint32_t i = 0; i < oob.tensor_ranges_count; ++i) {
        uint64_t packed = oob.tensor_ranges[i];
        uint32_t r_start = static_cast<uint32_t>(packed >> 32);
        uint32_t r_end = static_cast<uint32_t>(packed);
        if (r_end <= r_start) {
            continue;
        }
        // Skip any range overlapping a globally-allocated CB: the kernel may write
        // anywhere in such a CB (e.g. sharded move copies its whole dst), and a
        // live range overlapping one is either the CB's own buffer or a stale extent
        // nested in it — indistinguishable from an authorized CB write. See §12.
        bool overlaps_persistent_cb = false;
        for (uint64_t cb : persistent_cb_ranges) {
            if (r_start < static_cast<uint32_t>(cb) && static_cast<uint32_t>(cb >> 32) < r_end) {
                overlaps_persistent_cb = true;
                break;
            }
        }
        if (overlaps_persistent_cb) {
            continue;
        }
        // Skip I/O tensors this kernel was handed (see above).
        if (io_arg_starts.count(r_start) != 0) {
            continue;
        }
        Snap snap;
        snap.packed = packed;
        snap.bytes.resize(r_end - r_start);
        std::memcpy(snap.bytes.data(), l1_data + r_start, r_end - r_start);
        snapshots_.push_back(std::move(snap));
    }
}

void ObjectIntentTracker::accumulate_resolved(
    const EmuleOobTensorState& oob, const uint64_t* resolved_log, uint32_t count) {
    // snapshots_ non-empty ⇒ single-kernel core (one fiber here): gating on it keeps
    // this append off multi-kernel cores, where concurrent inserts would race.
    if (!oob.object_intent_strict || snapshots_.empty() || count == 0) {
        return;
    }
    resolved_acc_.insert(resolved_acc_.end(), resolved_log, resolved_log + count);
}

void ObjectIntentTracker::verify_post_launch(
    const uint8_t* l1_data, uint32_t lx, uint32_t ly, const char* kernel_name) const {
    if (snapshots_.empty()) {
        return;
    }
    std::unordered_set<uint64_t> resolved_set(resolved_acc_.begin(), resolved_acc_.end());
    for (const auto& snap : snapshots_) {
        if (resolved_set.count(snap.packed)) {
            continue;
        }
        uint32_t r_start = static_cast<uint32_t>(snap.packed >> 32);
        uint32_t r_end = static_cast<uint32_t>(snap.packed);
        if (std::memcmp(snap.bytes.data(), l1_data + r_start, r_end - r_start) != 0) {
            // No source line: detected post-exit by memcmp (the stray write bypassed
            // __emule_local_l1_to_ptr, so there is no captured call site). Kernel + core +
            // clobbered range are the actionable info; the cause is typically an overrun
            // from an adjacent buffer this kernel *did* resolve.
            __emule_asan_panic(
                "[ASAN ERROR] Object Intent Violation: Attempted to modify memory belonging to an "
                "adjacent object context — kernel %s on core (%u, %u) changed L1 buffer [0x%x, 0x%x) "
                "without ever resolving a pointer into it via __emule_local_l1_to_ptr (likely an overrun "
                "from an adjacent buffer). No source line: detected post-exit by memory comparison, after "
                "the kernel returned.\n",
                kernel_name ? kernel_name : "(unknown)",
                lx,
                ly,
                r_start,
                r_end);
        }
    }
}

void set_sanitizer_thread_locals(const EmuleOobTensorState& oob, uint32_t sem_base, uint32_t sem_size) {
    __emule_sem_l1_range_start = oob.asan_enabled ? sem_base : 0;
    __emule_sem_l1_range_end = oob.asan_enabled ? (sem_base + sem_size) : 0;
    __emule_l1_unreserved_base = oob.l1_unreserved_base;
    __emule_l1_tensor_ranges = oob.tensor_ranges;
    __emule_l1_tensor_ranges_count = oob.tensor_ranges_count;
    __emule_dram_unreserved_base = oob.dram_unreserved_base;
    __emule_dram_tensor_ranges = oob.dram_tensor_ranges;
    __emule_dram_tensor_ranges_count = oob.dram_tensor_ranges_count;
    __emule_l1_padding_ranges = oob.l1_padding_ranges;
    __emule_l1_padding_ranges_count = oob.l1_padding_ranges_count;
    __emule_l1_host_ranges = oob.l1_host_ranges;
    __emule_l1_host_ranges_count = oob.l1_host_ranges_count;
    __emule_cb_boundary_strict = oob.cb_boundary_strict;
}

void clear_sanitizer_thread_locals() {
    __emule_sem_l1_range_start = 0;
    __emule_sem_l1_range_end = 0;
    __emule_l1_unreserved_base = 0;
    __emule_l1_tensor_ranges = nullptr;
    __emule_l1_tensor_ranges_count = 0;
    __emule_dram_unreserved_base = 0;
    __emule_dram_tensor_ranges = nullptr;
    __emule_dram_tensor_ranges_count = 0;
    __emule_l1_padding_ranges = nullptr;
    __emule_l1_padding_ranges_count = 0;
    __emule_l1_host_ranges = nullptr;
    __emule_l1_host_ranges_count = 0;
    for (uint32_t i = 0; i < EMULE_NUM_CBS; ++i) {
        __emule_cb_reserved_pages[i] = 0;
        __emule_cb_waited_pages[i] = 0;
        __emule_cb_reserve_dangling[i] = false;
        __emule_cb_wait_dangling[i] = false;
        __emule_cb_reserve_file[i] = nullptr;
        __emule_cb_reserve_line[i] = 0;
        __emule_cb_wait_file[i] = nullptr;
        __emule_cb_wait_line[i] = 0;
    }
    __emule_cb_boundary_strict = false;
}

namespace {
void abort_if_dirty_cb(
    uint32_t cb_id,
    uint32_t unpushed,
    uint32_t unpopped,
    uint32_t lx,
    uint32_t ly,
    uint32_t processor_id,
    const char* reserve_file,
    uint32_t reserve_line,
    const char* wait_file,
    uint32_t wait_line) {
    // The kernel has already returned, so the offending file:line comes from the
    // call site captured at reserve/wait time. Only the imbalanced side(s) are reported.
    char reserve_clause[512] = "";
    if (unpushed > 0) {
        std::snprintf(
            reserve_clause,
            sizeof(reserve_clause),
            " %u page(s) reserved via cb_reserve_back at %s:%u were never committed with cb_push_back.",
            unpushed,
            reserve_file ? reserve_file : "?",
            reserve_line);
    }
    char wait_clause[512] = "";
    if (unpopped > 0) {
        std::snprintf(
            wait_clause,
            sizeof(wait_clause),
            " %u page(s) waited via cb_wait_front at %s:%u were never released with cb_pop_front.",
            unpopped,
            wait_file ? wait_file : "?",
            wait_line);
    }
    __emule_asan_panic(
        "[ASAN ERROR] Dirty CB Detected: Core (%u, %u) CB %u was not flushed! Kernel (processor %u):%s%s "
        "A cb_reserve_back with no following cb_push_back (or cb_wait_front with no following cb_pop_front) "
        "before the kernel exits leaves data the consumer is never signaled for; on silicon its matching "
        "cb_wait_front then hangs. (Lookahead producers that reserve more than they push but always push "
        "after their last reserve are not flagged.)\n",
        lx,
        ly,
        cb_id,
        processor_id,
        reserve_clause,
        wait_clause);
}
}  // namespace

// Fires only on a trailing dangling reserve/wait (one that no push/pop followed),
// using the per-CB dangling flags — deliberately decoupled from the cumulative
// window counters so lookahead producers aren't false-flagged. Full rationale,
// the lookahead example, and the known trade-off are in SANITIZER_CHECKS.md §11.
void sweep_per_kernel_dirty_cbs(
    const EmuleOobTensorState& oob, tt_emule::CBSyncState* cb_array, uint32_t processor_id, uint32_t lx, uint32_t ly) {
    if (!oob.asan_enabled || cb_array == nullptr) {
        return;
    }
    // Per-check opt-out (TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB); see host_sanitizers.hpp.
    if (dirty_cb_check_skipped()) {
        return;
    }
    for (uint32_t cb_id = 0; cb_id < EMULE_NUM_CBS; ++cb_id) {
        if (cb_array[cb_id].num_pages == 0) {
            continue;
        }
        // The reported page count is the window counter, which for a genuine
        // dangling reserve is the unpushed amount.
        uint32_t unpushed = __emule_cb_reserve_dangling[cb_id] ? __emule_cb_reserved_pages[cb_id] : 0;
        uint32_t unpopped = __emule_cb_wait_dangling[cb_id] ? __emule_cb_waited_pages[cb_id] : 0;
        if (unpushed > 0 || unpopped > 0) {
            abort_if_dirty_cb(
                cb_id,
                unpushed,
                unpopped,
                lx,
                ly,
                processor_id,
                __emule_cb_reserve_file[cb_id],
                __emule_cb_reserve_line[cb_id],
                __emule_cb_wait_file[cb_id],
                __emule_cb_wait_line[cb_id]);
        }
    }
}

OobStateOwner build_oob_tensor_state(IDevice* device, int device_id) {
    OobStateOwner owner;
    const bool asan = emule_asan_enabled();
    owner.state.asan_enabled = asan;
    owner.state.cb_boundary_strict = asan;
    if (!asan) {
        return owner;
    }
    static const uint64_t kEmptyRange = 0;

    owner.live_ranges = tt::tt_metal::emule::LiveL1Ranges::snapshot(device_id);
    owner.state.l1_unreserved_base =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    owner.state.tensor_ranges = owner.live_ranges.empty() ? &kEmptyRange : owner.live_ranges.data();
    owner.state.tensor_ranges_count = static_cast<uint32_t>(owner.live_ranges.size());

    owner.dram_live_ranges = tt::tt_metal::emule::LiveDramRanges::snapshot(device_id);
    owner.state.dram_unreserved_base =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::DRAM));
    owner.state.dram_tensor_ranges = owner.dram_live_ranges.empty() ? &kEmptyRange : owner.dram_live_ranges.data();
    owner.state.dram_tensor_ranges_count = static_cast<uint32_t>(owner.dram_live_ranges.size());
    owner.state.object_intent_strict = true;

    owner.padding_ranges = tt::tt_metal::emule::LiveL1PaddingRanges::snapshot(device_id);
    if (!owner.padding_ranges.empty()) {
        owner.state.l1_padding_ranges = owner.padding_ranges.data();
        owner.state.l1_padding_ranges_count = static_cast<uint32_t>(owner.padding_ranges.size());
    }

    owner.l1_host_ranges = tt::tt_metal::emule::LiveL1HostPokeRanges::snapshot(device_id);
    if (!owner.l1_host_ranges.empty()) {
        owner.state.l1_host_ranges = owner.l1_host_ranges.data();
        owner.state.l1_host_ranges_count = static_cast<uint32_t>(owner.l1_host_ranges.size());
    }
    return owner;
}

}  // namespace tt::tt_metal::emule
