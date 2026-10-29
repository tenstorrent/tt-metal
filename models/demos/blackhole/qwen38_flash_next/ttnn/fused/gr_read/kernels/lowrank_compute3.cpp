// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lowrank_compute.cpp for the GR gate v3 (QWEN38_GR_GATE_V3=1, rows <= 16): the same fold and typecast; silu and the
// injection sigmoid on rows 0..15 only (VectorMode::R), which holds every valid row.  The staged and low-rank tiles'
// rows 16..31 are not computed.
#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t D = get_compile_time_arg_val(1);
    constexpr uint32_t inject = get_compile_time_arg_val(2);
    constexpr uint32_t c_pg = 0;
    constexpr uint32_t c_zero = 1;
    constexpr uint32_t c_q = 2;
    constexpr uint32_t c_lr = get_compile_time_arg_val(3);
    constexpr uint32_t c_inj = get_compile_time_arg_val(4);
    constexpr uint32_t fp32 = static_cast<uint32_t>(DataFormat::Float32);
    constexpr uint32_t bf16 = static_cast<uint32_t>(DataFormat::Float16_b);

    compute_kernel_hw_startup(c_pg, c_zero, c_q);
    DataflowBuffer pg(c_pg);
    DataflowBuffer zero(c_zero);
    DataflowBuffer q(c_q);
    DataflowBuffer lr(c_lr);
    DataflowBuffer inj(c_inj);

    {
        FUSED_ZONE("fz_gr_lr_c_fold");
        zero.wait_front(1);
        pg.wait_front(T * D);
        for (uint32_t t = 0; t < T; ++t) {
            add_init(c_pg, c_zero, true);
            reconfig_data_format(c_pg, c_zero);
            tile_regs_acquire();
            for (uint32_t d = 0; d < D; ++d) {
                add_tiles(c_pg, c_zero, t * D + d, 0, 0);
            }
            typecast_tile_init<fp32, bf16>();
            typecast_tile<fp32, bf16>(0);
            tile_regs_commit();
            q.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_q);
            pack_tile(0, c_q);
            tile_regs_release();
            q.push_back(1);
        }
        pg.pop_front(T * D);
        zero.pop_front(1);
    }

    {
        FUSED_ZONE("fz_gr_lr_c_silu");
        q.wait_front(T);
        reconfig_data_format_srca(c_pg, c_q);
        copy_init(c_q);
        silu_tile_init();
        for (uint32_t t = 0; t < T; ++t) {
            tile_regs_acquire();
            copy_tile(c_q, t, 0);
            MATH(
                (SFPU_UNARY_CALL(DST_SYNC_MODE, false, calculate_silu, (false, 8 /* ITERATIONS */), 0, VectorMode::R)));
            tile_regs_commit();
            lr.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_lr);
            pack_tile(0, c_lr);
            tile_regs_release();
            lr.push_back(1);
        }
        if constexpr (inject < T) {
            tile_regs_acquire();
            copy_tile(c_q, inject, 0);
            sigmoid_tile_init<false>();
            sigmoid_tile<VectorMode::R, false, false>(0);
            binop_with_scalar_tile_init();
            MATH((SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                true,
                calculate_binop_with_scalar,
                (APPROX, MUL_UNARY, 8 /* ITERATIONS */, true),
                0,
                VectorMode::R,
                0x40000000u)));
            tile_regs_commit();
            inj.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_inj);
            pack_tile(0, c_inj);
            tile_regs_release();
            inj.push_back(1);
        }
        q.pop_front(T);
    }
}
