// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "compute_math.hpp"
#include "api/compute/sfpu_binary_bcast.h"

// Experimental register-resident value-column partition. SyncFull provides
// eight FP32 DEST tiles: four state tiles (4..7), delta (0), operand (1),
// spare (2), and the ordered reduction accumulator (3). No state is narrowed,
// and the reader/writer retain the original disjoint DRAM ownership.
void kernel_main() {
    constexpr uint32_t value_columns = get_compile_time_arg_val(0);
    constexpr bool normalize_qk = get_compile_time_arg_val(1) != 0;
    static_assert(value_columns == 1 && !normalize_qk);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 2, 8);
    for (uint32_t item = 0; item < count; ++item) {
        cb_wait_front(0, 4);
        cb_wait_front(1, 4);
        cb_wait_front(2, 1);
        cb_wait_front(3, 1);
        cb_wait_front(4, 1);
        cb_wait_front(5, 4);
        cb_reserve_back(7, 4);
        cb_reserve_back(8, 1);
        tile_regs_acquire();

        // delta = beta * (v - k^T (decay * S)). Retain each decayed
        // FP32 state tile instead of recomputing it in the update pass.
        for (uint32_t kr = 0; kr < 4; ++kr) {
            const uint32_t state = 4 + kr;
            const uint32_t product = kr == 0 ? 3 : 0;
            load(5, kr, state);
            broadcast<BroadcastType::SCALAR>(3, 0, 1);
            multiply(state, 1, state);
            broadcast<BroadcastType::COL>(1, kr, 1);
            multiply(state, 1, product);
            if (kr != 0) {
                add(3, 0, 3);
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
        load(2, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(1, 3, 0);
        broadcast<BroadcastType::SCALAR>(4, 0, 1);
        multiply(0, 1, 0);

        // The reduction's row zero holds delta. Broadcast it inside DEST,
        // preserving the existing separate FP32 multiply/add (no FMA).
        for (uint32_t kr = 0; kr < 4; ++kr) {
            broadcast<BroadcastType::COL>(1, kr, 1);
            sfpu_mul_bcast_row_init();
            sfpu_mul_bcast_row(1, 0);
            add(4 + kr, 1, 4 + kr);
        }

        // Output = q^T S_new. Consume the retained updated state directly;
        // the order of the four partial sums and column reduction is unchanged.
        for (uint32_t kr = 0; kr < 4; ++kr) {
            const uint32_t product = kr == 0 ? 3 : 0;
            broadcast<BroadcastType::COL>(0, kr, 1);
            multiply(4 + kr, 1, product);
            if (kr != 0) {
                add(3, 0, 3);
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format<true>(7);
        pack_init(7);
        for (uint32_t kr = 0; kr < 4; ++kr) {
            pack_tile<true>(4 + kr, 7, kr);
        }
        pack_reconfig_data_format<true>(8);
        pack_init(8);
        pack_tile<true>(3, 8, 0);
        tile_regs_release();
        cb_push_back(7, 4);
        cb_push_back(8, 1);
        cb_pop_front(0, 4);
        cb_pop_front(1, 4);
        cb_pop_front(2, 1);
        cb_pop_front(3, 1);
        cb_pop_front(4, 1);
        cb_pop_front(5, 4);
        // CB7/8 retain the writer as their sole consumer. CB6 is unused.
    }
}
