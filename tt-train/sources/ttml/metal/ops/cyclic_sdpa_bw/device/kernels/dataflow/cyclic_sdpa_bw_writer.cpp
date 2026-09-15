// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Results of the cyclic backward pass, from L1 back to DRAM, and the
// chip-wide barrier that orders them.
//
// All three gradients are written before this core arrives, so the release of
// t certifies both that every core's dQ write for t has completed and that
// this core's own column writes have -- which is what lets the reader load
// all three seeds after waiting on it.
//
// The causal mask tile is generated here, once, as sdpa_bw's writer does.
//
// With one core the release is written locally rather than multicast: a
// multicast whose destination rectangle is just the sender never completes,
// and the writer parks in its write barrier while the reader waits for a
// release that never arrives. C = 1 is the smallest complete schedule --
// three pairs, one diagonal, and a column revisit -- so it is worth a branch.
//
// Degenerate rectangles matter on the other axis too: a destination rectangle
// one row high trips a device-side assert, while one column wide is fine.
// Choose the region's shape accordingly -- the serpentine embedding keeps
// every snake edge at one hop for either orientation.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef FOLD_SCALE_INTO_KEY
#define FOLD_SCALE_INTO_KEY 0
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(arg++);
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
    // Row-tiles per block: B = Bt * 32, so a block is Bt * qWt tiles wide in
    // memory and the statistics are Bt tiles instead of one.
    constexpr uint32_t Bt = get_compile_time_arg_val(5);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;
    constexpr auto grad_query_args = TensorAccessorArgs<6>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_17;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_20;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_23;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;
    // The statistics the reader loads, one slot, and the tiles made of them.
    constexpr uint32_t cb_lse = tt::CBIndex::c_4;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
    constexpr uint32_t cb_lse_row = tt::CBIndex::c_13;      // -L, row layout
    constexpr uint32_t cb_u_row = tt::CBIndex::c_14;        // -D, row layout
    constexpr uint32_t cb_lse_rem_row = tt::CBIndex::c_30;  // -L's remainder, Float32 row
    constexpr uint32_t cb_lse_rem = tt::CBIndex::c_9;       // -L's remainder, bfloat16 column
    constexpr uint32_t cb_u_rem = tt::CBIndex::c_29;        // -D's remainder, bfloat16 column

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;

    // The compute kernel forms S^T, so its diagonal tile takes the transposed
    // causal mask, in additive form (0 or -inf): live where the key index is
    // at most the query index.
    cyclic_dataflow::generate_causal_mask_tiles(cb_attn_mask);
    // The compute kernel adds the mask with an accumulating FPU add whose
    // second operand is this zero tile: the fence buffer's page, never
    // written by anyone else (only its push and pop counters are used).
    cyclic_dataflow::zero_tile(get_write_ptr(tt::CBIndex::c_8), get_tile_size(tt::CBIndex::c_8));
    cyclic_dataflow::generate_ones_column_tile(tt::CBIndex::c_28);  // for the D remainder

    const uint32_t grad_bytes = get_tile_size(cb_grad_query);
    const auto grad_query = TensorAccessor(grad_query_args, grad_query_addr, grad_bytes);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);

    volatile tt_l1_ptr uint32_t* arrive_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_sem_id));
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    const uint32_t interm_bytes = get_tile_size(cb_lse);
    const uint32_t base_lse = get_write_ptr(cb_lse);  // one slot: always here
    const uint32_t base_u_scalar = get_write_ptr(cb_u_scalar);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_lse_row), 2u * Bt * interm_bytes);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_u_row), 2u * Bt * interm_bytes);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_lse_rem_row), 2u * Bt * interm_bytes);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_lse_rem), 2u * Bt * get_tile_size(cb_lse_rem));
    cyclic_dataflow::zero_tile(get_write_ptr(cb_u_rem), 2u * Bt * get_tile_size(cb_u_rem));
    const uint64_t arrive_noc_addr = get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(arrive_sem_id));
    const uint64_t release_mcast_addr = get_noc_multicast_addr(
        mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, get_semaphore(release_sem_id));

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);

        // The statistic tiles for this timestep, once the reader has L and D.
        cb_wait_front(cyclic_dataflow::kStatsReadyCb, 1);
        invalidate_l1_cache();
        cyclic_dataflow::produce_statistic_tiles<FOLD_SCALE_INTO_KEY != 0>(
            base_lse, base_u_scalar, Bt, interm_bytes, cb_lse_row, cb_u_row, cb_lse_rem_row, cb_lse_rem, cb_u_rem);
        cb_pop_front(cyclic_dataflow::kStatsReadyCb, 1);

        write_tiles_by_row(cb_grad_query, grad_query, (pair.i - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
        write_tiles_by_row(cb_grad_key, grad_key, (pair.j - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
        write_tiles_by_row(cb_grad_value, grad_value, (pair.j - 1u) * val_tiles, val_tiles, grad_bytes, val_tiles);

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
    }
}
