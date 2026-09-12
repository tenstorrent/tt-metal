// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Column gradients and the chip-wide barrier, for the relay variant.
//
// The row packet is the relay reader's business: it receives it, hands it to
// the compute kernel and forwards it. What is left here is the column side,
// which still goes through DRAM every timestep, and the barrier that orders a
// streak-start reload after the preceding streak's spill.
//
// The mask tile is generated here, once, as sdpa_bw's writer does.
//
// ENDPOINT_SYNC selects Algorithm 4: no chip-wide barrier at all. What the
// barrier did here was tell this core's reader that the column gradients it
// is about to load have been written -- an ordering between two RISCs of the
// same core, which needs no chip-wide anything. A local progress word does
// it. The other thing the barrier did, ordering a streak-start reload after
// the preceding streak's spill, is the reader's business and becomes the
// endpoint counters.
//
// The column gradients only pass through DRAM at all because this step has
// not yet restored the paper's column residency; with them resident, this
// handoff disappears too.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"

#ifndef ENDPOINT_SYNC
#define ENDPOINT_SYNC 0
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_end = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_end = get_arg_val<uint32_t>(arg++);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(4);
    constexpr auto grad_key_args = TensorAccessorArgs<5>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_20;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_23;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_25;
    constexpr uint32_t cb_column_progress = tt::CBIndex::c_26;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;

    generate_causal_mask_tile(cb_attn_mask);

    const uint32_t grad_bytes = get_tile_size(cb_grad_key);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);

#if ENDPOINT_SYNC
    volatile tt_l1_ptr uint32_t* column_progress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_column_progress));
    *column_progress = 0u;
#endif

    volatile tt_l1_ptr uint32_t* arrive_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_sem_id));
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    const uint64_t arrive_noc_addr = get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(arrive_sem_id));
    const uint64_t release_mcast_addr = get_noc_multicast_addr(
        mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, get_semaphore(release_sem_id));

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);

        // The column gradients are handed over once per residency interval,
        // at its end -- which the schedule says is a column change or the
        // last timestep. Writing them back before reusing their storage is
        // the paper's rule; here the storage is released by the handover
        // itself.
        const bool column_ends =
            (t + 1u == kTimesteps) || (sched.pair(my_core, t + 1u).j != pair.j);
        if (column_ends) {
            write_tiles_by_row(cb_grad_value, grad_value, (pair.j - 1u) * vWt, vWt, grad_bytes, vWt);
            write_tiles_by_row(cb_grad_key, grad_key, (pair.j - 1u) * qWt, qWt, grad_bytes, qWt);
        }

#if ENDPOINT_SYNC
        // This core's own reader is the only thing waiting on these writes.
        *column_progress = t + 1u;
#else
        noc_semaphore_inc(arrive_noc_addr, 1u);

        if (is_coordinator != 0u) {
            WAYPOINT("ARVW");
            do {
                invalidate_l1_cache();
            } while ((*arrive_sem) < kCores * (t + 1u));
            WAYPOINT("ARVD");
            if constexpr (kCores == 1u) {
                volatile tt_l1_ptr uint32_t* release_local =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
                noc_semaphore_set(release_local, t + 1u);
            } else {
                *scratch = t + 1u;
                noc_semaphore_set_multicast_loopback_src(scratch_l1, release_mcast_addr, kCores);
                noc_async_write_barrier();
            }
        }
#endif
    }
}
