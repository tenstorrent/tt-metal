// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Operands of the cyclic backward pass, from DRAM into L1, one timestep at a
// time: the row packet (Q_i, dO_i, L_i, D_i), the column state (K_j, V_j),
// and the three gradients this pair adds to.
//
// Two orderings matter and the barrier provides both. dQ_i must carry every
// update from earlier timesteps, and other cores produce those. dK_j and dV_j
// are this core's alone -- each column has one owner -- but they were last
// written by this core's *writer*, which runs on another RISC and could still
// be behind. Waiting for the release of t - 1 covers both, because the writer
// arrives only after all three of its writes have completed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t u_scalar_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(3);
    // Row-tiles per block: B = Bt * 32, so a block is Bt * qWt tiles wide in
    // memory and the statistics are Bt tiles instead of one.
    constexpr uint32_t Bt = get_compile_time_arg_val(4);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;
    constexpr auto query_args = TensorAccessorArgs<5>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto grad_output_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<lse_args.next_compile_time_args_offset()>();
    constexpr auto grad_query_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_lse = tt::CBIndex::c_4;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
    constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
    constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
    constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const uint32_t interm_bytes = get_tile_size(cb_lse);
    const uint32_t grad_bytes = get_tile_size(cb_grad_query_seed);

    const auto query = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto grad_output = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto lse = TensorAccessor(lse_args, lse_addr, interm_bytes);
    const auto u_scalar = TensorAccessor(u_args, u_scalar_addr, interm_bytes);
    const auto grad_query = TensorAccessor(grad_query_args, grad_query_addr, grad_bytes);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);

    volatile tt_l1_ptr uint32_t* release_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const uint32_t i = pair.i;
        const uint32_t j = pair.j;

        read_tiles_by_row(cb_query, query, (i - 1u) * row_tiles, row_tiles, tile_bytes, row_tiles);
        read_tiles_by_row(cb_key, key, (j - 1u) * row_tiles, row_tiles, tile_bytes, row_tiles);
        read_tiles_by_row(cb_value, value, (j - 1u) * val_tiles, val_tiles, tile_bytes, val_tiles);
        read_tiles_by_row(cb_grad_output, grad_output, (i - 1u) * val_tiles, val_tiles, tile_bytes, val_tiles);
        read_tiles_by_row(cb_lse, lse, (i - 1u) * Bt, Bt, interm_bytes, Bt);
        read_tiles_by_row(cb_u_scalar, u_scalar, (i - 1u) * Bt, Bt, interm_bytes, Bt);

        if (t > 0u) {
            WAYPOINT("BARW");
            do {
                invalidate_l1_cache();
            } while ((*release_sem) < t);
            WAYPOINT("BARD");
        }

        read_tiles_by_row(cb_grad_query_seed, grad_query, (i - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
        read_tiles_by_row(cb_grad_key_seed, grad_key, (j - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
        read_tiles_by_row(cb_grad_value_seed, grad_value, (j - 1u) * val_tiles, val_tiles, grad_bytes, val_tiles);
    }
}
