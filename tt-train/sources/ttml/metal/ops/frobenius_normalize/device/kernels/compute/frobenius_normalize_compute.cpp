// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Frobenius normalize compute kernel — all-FP32 data path.
//
// Phase 1:   square + accumulate (SFPU FP32 in DST)
// Phase 2:   sfpu_reduce (FP32 in DST) — within-tile sum-of-squares
// Phase 3/4: copy_tile (UnpackToDestFp32) + add_binary_tile (SFPU FP32) — chain reduction
// Phase 5:   copy_tile (UnpackToDestFp32) + sqrt/recip (SFPU FP32) — origin norm compute
// Phase 6:   copy_tile (UnpackToDestFp32) + mul_binary_tile (SFPU FP32) — normalize
//
// The only BF16 truncation points are input unpack and output pack.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);

constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_sq_acc = tt::CBIndex::c_1;
constexpr auto cb_scalar = tt::CBIndex::c_2;
constexpr auto cb_recv = tt::CBIndex::c_3;
constexpr auto cb_norm = tt::CBIndex::c_4;
constexpr auto cb_output = tt::CBIndex::c_5;

void kernel_main() {
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
            mul_binary_tile(reg, reg, reg);
            if (i > 0) {
                add_binary_tile_init();
                add_binary_tile(accum_reg, work_reg, accum_reg);
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
    // Phase 2: sfpu_reduce — within-tile reduction to scalar (FP32 in DST)
    // =========================================================================
    {
        cb_wait_front(cb_sq_acc, 1);
        cb_reserve_back(cb_scalar, 1);

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
    // Phase 3: Row chain add (FP32: copy_tile UnpackToDestFp32 + add_binary_tile SFPU)
    // =========================================================================
    if (do_row_receive) {
        cb_wait_front(cb_recv, 1);
        cb_wait_front(cb_scalar, 1);
        cb_reserve_back(cb_scalar, 1);
        tile_regs_acquire();
        copy_tile_init(cb_scalar);
        copy_tile(cb_scalar, 0, 0);
        copy_tile_init(cb_recv);
        copy_tile(cb_recv, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
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
    // Phase 4: Column chain add (same as Phase 3)
    // =========================================================================
    if (do_col_receive) {
        cb_wait_front(cb_recv, 1);
        cb_wait_front(cb_scalar, 1);
        cb_reserve_back(cb_scalar, 1);
        tile_regs_acquire();
        copy_tile_init(cb_scalar);
        copy_tile(cb_scalar, 0, 0);
        copy_tile_init(cb_recv);
        copy_tile(cb_recv, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
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
    // Phase 5: Origin: sqrt + eps + recip → pack to cb_norm
    // =========================================================================
    if (is_origin) {
        cb_wait_front(cb_scalar, 1);
        cb_reserve_back(cb_norm, 1);

        tile_regs_acquire();
        copy_tile_init(cb_scalar);
        copy_tile(cb_scalar, 0, 0);

        sqrt_tile_init();
        sqrt_tile(0);

        // eps tile from reader in cb_sq_acc (repurposed)
        cb_wait_front(cb_sq_acc, 1);
        copy_tile_init(cb_sq_acc);
        copy_tile(cb_sq_acc, 0, 1);
        cb_pop_front(cb_sq_acc, 1);

        add_binary_tile_init();
        add_binary_tile(0, 1, 0);

        recip_tile_init();
        recip_tile(0);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_norm);
        pack_tile(0, cb_norm);
        tile_regs_release();
        cb_push_back(cb_norm, 1);
        cb_pop_front(cb_scalar, 1);
    }

    // =========================================================================
    // Phase 6: Multiply each tile by 1/norm (FP32: mul_binary_tile SFPU)
    //
    // cb_norm has 1/norm in EVERY position (reader filled it via multicast scalar).
    // We reload cb_norm into DST each iteration since tile_regs_release clears DST.
    // =========================================================================
    // Read the FP32 1/norm scalar from cb_norm tile position (0,0) via mailbox
    cb_wait_front(cb_norm, 1);
    uint32_t norm_u32 = read_tile_value(cb_norm, 0, 0);
    cb_pop_front(cb_norm, 1);

    // Reinit for BF16 input after Phases 2-5 configured for FP32 CBs
    init_sfpu(cb_input, cb_output);

    for (uint32_t i = 0; i < num_tiles_per_core; ++i) {
        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);

        tile_regs_acquire();
        copy_tile_init(cb_input);
        copy_tile(cb_input, 0, 0);
        cb_pop_front(cb_input, 1);

        // FP32 scalar multiply: every element *= 1/norm (SFPU, full FP32 in DST)
        binop_with_scalar_tile_init();
        mul_unary_tile(0, norm_u32);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_output);
        pack_tile(0, cb_output);
        tile_regs_release();

        cb_push_back(cb_output, 1);
    }
}
