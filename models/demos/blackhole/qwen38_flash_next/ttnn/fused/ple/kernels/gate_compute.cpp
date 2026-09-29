// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE stage 3 (`Qwen38TTNNPLE._gate` after the two norm all_gathers), one core, 32-bit dest: the chain's ops in the
// chain's order, each exactly the LLK its program runs, the fp32 intermediates kept in the dest or packed fp32 and
// re-read exact (the chain's fp32 tensors between programs).
//   typecast(key), typecast(query) -> bf16 unpacked into the fp32 dest (exact), multiply fp32 (mul_binary_tile),
//   packed fp32 (Wt tiles);  sum(dim 3) -> the accurate fp32 SFPU fold (compute_kernel_lib::reduce REDUCE_ROW, Ht 1 x
//   Wt, Accurate);  multiply by 2560^-0.5 -> mul_binary_tile with the scalar-filled fp32 tile (binary_ng's SFPU scalar
//   kernel);  abs_tile;  clamp_tile(1e-6, FLT_MAX) (clamp_tss);  sqrt_tile<false>;  sign_tile of the kept gate;
//   multiply (mul_binary_tile);  sigmoid_tile<VectorMode::RC, 0u> (accurate);  then per column tile of the value row
//   repeated over the branch rows: unary_bcast<COL> of the coefficient column, packed fp32 and re-read, times the
//   value tile (bf16 exact in the dest), packed bf16 (binary_ng's SFPU col-bcast kernel, bf16 out).
// CBs: 0 key (bf16, Wt), 1 query (bf16, Wt), 2 products (fp32, Wt), 3 reduce scaler (fp32, 1; waited, ignored),
// 4 sum (fp32, 1), 5 scale tile (fp32, 1), 6 gate copy (fp32, 1), 7 coefficient (fp32, 1), 8 bcast (fp32, 1),
// 9 value branches (bf16, Vt), 16 gated (bf16, Vt).  Compile-time args: 0 Wt (80), 1 Vt (20), 2 debug (1: also pack
// the sum and coefficient tiles to CBs 17 / 18 for the writer).
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(2);
    constexpr uint32_t c_key = 0, c_query = 1, c_prod = 2, c_scaler = 3, c_sum = 4, c_scale = 5, c_gate = 6, c_coef = 7,
                       c_bc = 8, c_val = 9, c_out = 16, c_dbg_sum = 17, c_dbg_coef = 18;
    constexpr uint32_t MIN_BITS = 0x358637bdu;  // 1e-6f
    constexpr uint32_t MAX_BITS = 0x7f7fffffu;  // FLT_MAX
    compute_kernel_hw_startup(c_key, c_query, c_prod);
    DataflowBuffer key(c_key), query(c_query), prod(c_prod), scaler(c_scaler), sum(c_sum), scale(c_scale), gate(c_gate),
        coef(c_coef), bc(c_bc), val(c_val), out(c_out);

    {
        FUSED_ZONE("fz_pl_gate_c_products");
        // 1. products: fp32(key) * fp32(query), packed fp32
        key.wait_front(Wt);
        query.wait_front(Wt);
        reconfig_data_format(c_key, c_query);
        pack_reconfig_data_format(c_prod);
        copy_init(c_key);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < Wt; ++t) {
            tile_regs_acquire();
            reconfig_data_format_srca(c_query, c_key);
            copy_init(c_key);
            copy_tile(c_key, t, 0);
            reconfig_data_format_srca(c_key, c_query);
            copy_init(c_query);
            copy_tile(c_query, t, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            prod.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, c_prod);
            tile_regs_release();
            prod.push_back(1);
        }
        key.pop_front(Wt);
        query.pop_front(Wt);
    }

    {
        FUSED_ZONE("fz_pl_gate_c_sum");
        // 2. the chain's ttnn.sum over W: the accurate fp32 SFPU fold (the helper waits on the scaler, pops the input)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            c_prod,
            c_scaler,
            c_sum,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Accurate>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});
        scaler.pop_front(1);
    }

    {
        FUSED_ZONE("fz_pl_gate_c_scalar");
        // 3. the scalar chain on the sum tile: gate = sum * scale; |gate|, clamp, sqrt; sign(gate); product; sigmoid
        sum.wait_front(1);
        scale.wait_front(1);
        reconfig_data_format(c_sum, c_scale);
        pack_reconfig_data_format(c_gate);
        tile_regs_acquire();
        copy_init(c_sum);
        copy_tile(c_sum, 0, 0);
        reconfig_data_format_srca(c_sum, c_scale);
        copy_init(c_scale);
        copy_tile(c_scale, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // gate (fp32)
        tile_regs_commit();
        gate.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_gate);
        if constexpr (DEBUG) {
            DataflowBuffer dbg_sum(c_dbg_sum);
            dbg_sum.reserve_back(1);
            pack_reconfig_data_format(c_dbg_sum);
            pack_tile(0, c_dbg_sum);
            dbg_sum.push_back(1);
            pack_reconfig_data_format(c_gate);
        }
        tile_regs_release();
        gate.push_back(1);
        sum.pop_front(1);
        scale.pop_front(1);

        gate.wait_front(1);
        reconfig_data_format_srca(c_scale, c_gate);
        pack_reconfig_data_format(c_coef);
        tile_regs_acquire();
        copy_init(c_gate);
        copy_tile(c_gate, 0, 0);  // magnitude chain on dst 0
        abs_tile_init();
        abs_tile(0);
        clamp_tile_init();
        clamp_tile(0, MIN_BITS, MAX_BITS);
        sqrt_tile_init();
        sqrt_tile<false>(0);
        copy_tile(c_gate, 0, 1);  // direction on dst 1
        sign_tile_init();
        sign_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // transformed
        sigmoid_tile_init<0u>();
        sigmoid_tile<VectorMode::RC, 0u>(0);  // coefficient
        tile_regs_commit();
        coef.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_coef);
        if constexpr (DEBUG) {
            DataflowBuffer dbg_coef(c_dbg_coef);
            dbg_coef.reserve_back(1);
            pack_reconfig_data_format(c_dbg_coef);
            pack_tile(0, c_dbg_coef);
            dbg_coef.push_back(1);
            pack_reconfig_data_format(c_coef);
        }
        tile_regs_release();
        coef.push_back(1);
        gate.pop_front(1);
    }

    {
        FUSED_ZONE("fz_pl_gate_c_gated");
        // 4. gated = value branches * coefficient (column broadcast), bf16 out: binary_ng's SFPU col-bcast kernel
        coef.wait_front(1);
        reconfig_data_format_srca(c_gate, c_coef);
        pack_reconfig_data_format(c_bc);
        unary_bcast_init<BroadcastType::COL>(c_coef);
        tile_regs_acquire();
        unary_bcast<BroadcastType::COL>(c_coef, 0, 0);
        tile_regs_commit();
        bc.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_bc);
        tile_regs_release();
        bc.push_back(1);
        coef.pop_front(1);

        bc.wait_front(1);
        val.wait_front(Vt);
        pack_reconfig_data_format(c_out);
        for (uint32_t t = 0; t < Vt; ++t) {
            tile_regs_acquire();
            reconfig_data_format_srca(c_bc, c_val);
            copy_init(c_val);
            copy_tile(c_val, t, 0);
            reconfig_data_format_srca(c_val, c_bc);
            copy_init(c_bc);
            copy_tile(c_bc, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            out.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, c_out);
            tile_regs_release();
            out.push_back(1);
        }
        val.pop_front(Vt);
        bc.pop_front(1);
    }
}
