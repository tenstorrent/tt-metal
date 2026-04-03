// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
// reduce.h not needed — using sfpu_reduce
#include "api/compute/tile_move_copy.h"

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);

constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_sq_acc = tt::CBIndex::c_1;
constexpr auto cb_scalar = tt::CBIndex::c_2;
constexpr auto cb_recv = tt::CBIndex::c_3;
constexpr auto cb_norm = tt::CBIndex::c_4;
constexpr auto cb_output = tt::CBIndex::c_5;
constexpr auto cb_scaler = tt::CBIndex::c_6;

void kernel_main() {
    // Runtime args for per-core role
    uint32_t rt = 0;
    uint32_t do_row_receive = get_arg_val<uint32_t>(rt++);
    uint32_t do_row_send = get_arg_val<uint32_t>(rt++);
    uint32_t do_col_receive = get_arg_val<uint32_t>(rt++);
    uint32_t do_col_send = get_arg_val<uint32_t>(rt++);
    uint32_t is_origin = get_arg_val<uint32_t>(rt++);

    constexpr uint32_t accum_reg = 0;
    constexpr uint32_t work_reg = 1;

    init_sfpu(cb_input, cb_output);
    binary_op_init_common(cb_input, cb_input, cb_output);

    // =========================================================================
    // Phase 1: Square and accumulate all input tiles into one FP32 tile
    // For cores with 0 tiles, reader generates a zero tile in cb_sq_acc.
    // =========================================================================
    if (num_tiles_per_core > 0) {
        cb_reserve_back(cb_sq_acc, 1);
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles_per_core; ++i) {
            cb_wait_front(cb_input, 1);

            auto reg = (i == 0) ? accum_reg : work_reg;
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, reg);

            mul_binary_tile_init();
            mul_binary_tile(reg, reg, reg);  // square

            if (i > 0) {
                add_binary_tile_init();
                add_binary_tile(accum_reg, work_reg, accum_reg);  // accumulate
            }

            cb_pop_front(cb_input, 1);
        }
        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_sq_acc);
        pack_tile(accum_reg, cb_sq_acc);
        tile_regs_release();
        cb_push_back(cb_sq_acc, 1);
    }
    // else: reader has generated a zero tile in cb_sq_acc for us

    // =========================================================================
    // Phase 2: Reduce 32x32 accumulated tile to a scalar tile
    // Uses sfpu_reduce for full FP32 precision in DST registers.
    // init_sfpu configures the unpack-to-dest path so copy_tile writes
    // directly to DST (required for SFPLOAD in sfpu_reduce to read data).
    // =========================================================================
    {
        cb_wait_front(cb_sq_acc, 1);
        cb_reserve_back(cb_scalar, 1);

        // Reinitialize for sfpu_reduce: unpack/math/pack for cb_sq_acc→cb_scalar
        init_sfpu(cb_sq_acc, cb_scalar);

        tile_regs_acquire();
        copy_tile_init(cb_sq_acc);
        copy_tile(cb_sq_acc, 0, 0);
        cb_pop_front(cb_sq_acc, 1);

        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_scalar);
        pack_tile(0, cb_scalar);
        tile_regs_release();
        cb_push_back(cb_scalar, 1);
    }

    // =========================================================================
    // Phase 3: Row chain add — receive from right neighbor and add
    // =========================================================================
    if (do_row_receive) {
        cb_wait_front(cb_recv, 1);
        cb_wait_front(cb_scalar, 1);

        cb_reserve_back(cb_scalar, 1);
        tile_regs_acquire();

        reconfig_data_format(cb_scalar, cb_recv);
        add_tiles_init(cb_scalar, cb_recv);
        add_tiles(cb_scalar, cb_recv, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_scalar);
        pack_tile(0, cb_scalar);
        tile_regs_release();

        cb_pop_front(cb_recv, 1);
        cb_pop_front(cb_scalar, 1);
        cb_push_back(cb_scalar, 1);
    }

    // Reader handles sending cb_scalar to left neighbor (if do_row_send).
    // Reader will pop cb_scalar after sending.

    // =========================================================================
    // Phase 4: Column chain add — receive from below and add (row leaders only)
    // =========================================================================
    if (do_col_receive) {
        cb_wait_front(cb_recv, 1);
        cb_wait_front(cb_scalar, 1);

        cb_reserve_back(cb_scalar, 1);
        tile_regs_acquire();

        reconfig_data_format(cb_scalar, cb_recv);
        add_tiles_init(cb_scalar, cb_recv);
        add_tiles(cb_scalar, cb_recv, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_scalar);
        pack_tile(0, cb_scalar);
        tile_regs_release();

        cb_pop_front(cb_recv, 1);
        cb_pop_front(cb_scalar, 1);
        cb_push_back(cb_scalar, 1);
    }

    // =========================================================================
    // Phase 5: Origin core computes recip(sqrt(sum) + eps) → cb_norm
    // =========================================================================
    if (is_origin) {
        cb_wait_front(cb_scalar, 1);
        cb_reserve_back(cb_norm, 1);

        tile_regs_acquire();
        copy_tile_init(cb_scalar);
        copy_tile(cb_scalar, 0, 0);

        sqrt_tile_init();
        sqrt_tile(0);

        // Add epsilon: we need the eps value. Reader generated an eps tile in cb_recv
        // (repurposed after chain reduction is done). Actually, reader puts eps in a
        // scratch area. Let's use a different approach: pass eps as a bfloat16 tile.
        // For simplicity, we'll use the scaler CB (c_6) which already has a ones tile,
        // and add eps via a separate mechanism.
        //
        // Actually, the simplest approach: reader generates an eps tile in cb_sq_acc
        // (reused, since phase 1 is done). But we can't control timing between reader
        // and compute for that.
        //
        // Best approach: add eps as a compile-time constant in the compute kernel.
        // We pack eps as bfloat16 and add it to the scalar value in register.
        // Since the scalar is in register after sqrt, we can add a constant.
        //
        // Use SFPU approach: the value in dest[0] is sqrt(sum). We need to add eps.
        // We'll copy the eps tile (generated by reader in cb_sq_acc, repurposed) and add.
        //
        // Simplest: reader pre-generates eps tile in cb_recv (after chain reduction
        // finishes, cb_recv is free). But ordering is tricky.
        //
        // Let's just use cb_sq_acc for eps tile. Reader generates it after phase 1
        // reads are done and before broadcast. We know cb_sq_acc is free after phase 2.
        // The reader can generate it right after the chain reduction completes.
        //
        // Actually, simplest safe approach: use a dedicated CB for eps. But we're
        // already using 7 CBs. Let me reuse cb_sq_acc (c_1) since it's consumed in
        // phase 2. Reader generates eps tile into cb_sq_acc for origin core only.

        // Wait for eps tile from reader (in cb_sq_acc, repurposed)
        cb_wait_front(cb_sq_acc, 1);
        copy_tile_init(cb_sq_acc);
        copy_tile(cb_sq_acc, 0, 1);
        cb_pop_front(cb_sq_acc, 1);

        add_binary_tile_init();
        add_binary_tile(0, 1, 0);  // sqrt(sum) + eps

        recip_tile_init();
        recip_tile(0);  // 1 / (sqrt(sum) + eps)

        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_norm);
        pack_tile(0, cb_norm);
        tile_regs_release();

        cb_push_back(cb_norm, 1);
        cb_pop_front(cb_scalar, 1);
    }

    // =========================================================================
    // Phase 6: Multiply each input tile by reciprocal norm
    // Reader re-reads tiles from DRAM into cb_input.
    // Reader/broadcast ensures cb_norm is filled on all cores.
    // =========================================================================
    cb_wait_front(cb_norm, 1);

    // Broadcast-scalar multiply: cb_norm has 1/norm at (0,0), broadcast to all elements
    reconfig_data_format(cb_input, cb_norm);
    mul_tiles_bcast_scalar_init_short(cb_input, cb_norm);

    for (uint32_t i = 0; i < num_tiles_per_core; ++i) {
        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);

        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_input, cb_norm, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_output);
        pack_tile(0, cb_output);
        tile_regs_release();

        cb_push_back(cb_output, 1);
        cb_pop_front(cb_input, 1);
    }

    cb_pop_front(cb_norm, 1);
}
