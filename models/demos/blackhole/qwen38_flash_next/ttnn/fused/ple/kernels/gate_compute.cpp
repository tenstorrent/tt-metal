// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gate (`Qwen38TTNNPLE._gate` after the two norm all_gathers), one core, 32-bit dest: the chain's ops in the
// chain's order, each exactly the LLK its program runs, the fp32 intermediates kept in the dest or packed fp32 and
// re-read exact (the chain's fp32 tensors between programs).
// CBs: 0 key (bf16, global_tiles), 1 query (bf16, global_tiles), 2 products (fp32, global_tiles), 3 reduce scaler
// (fp32, 1; waited, ignored), 4 sum (fp32, 1), 5 scale tile (fp32, 1), 6 gate copy (fp32, 1), 7 coefficient (fp32, 1),
// 8 broadcast (fp32, 1), 9 value branches (bf16, local_tiles), 16 gated (bf16, local_tiles).  Compile-time args:
// 0 global_tiles (the builder's GLOBAL_TILES, 80), 1 local_tiles (LOCAL_TILES, 20), 2 debug (1: also pack the sum and
// coefficient tiles to CBs 17 / 18 for the writer).
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

void kernel_main() {
    constexpr uint32_t global_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t local_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t debug = get_compile_time_arg_val(2);
    constexpr uint32_t cb_key = 0, cb_query = 1, cb_products = 2, cb_scaler = 3, cb_sum = 4, cb_scale = 5, cb_gate = 6,
                       cb_coefficient = 7, cb_broadcast = 8, cb_value = 9, cb_out = 16, cb_debug_sum = 17,
                       cb_debug_coefficient = 18;
    constexpr uint32_t clamp_min_bits = 0x358637bdu;  // 1e-6f
    constexpr uint32_t clamp_max_bits = 0x7f7fffffu;  // FLT_MAX
    compute_kernel_hw_startup(cb_key, cb_query, cb_products);
    DataflowBuffer key(cb_key), query(cb_query), products(cb_products), scaler(cb_scaler), sum(cb_sum), scale(cb_scale),
        gate(cb_gate), coefficient(cb_coefficient), broadcast(cb_broadcast), value(cb_value), out(cb_out);

    {
        // 1. products: fp32(key) * fp32(query), packed fp32
        key.wait_front(global_tiles);
        query.wait_front(global_tiles);
        reconfig_data_format(cb_key, cb_query);
        pack_reconfig_data_format(cb_products);
        copy_init(cb_key);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < global_tiles; ++t) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_query, cb_key);
            copy_init(cb_key);
            copy_tile(cb_key, t, 0);
            reconfig_data_format_srca(cb_key, cb_query);
            copy_init(cb_query);
            copy_tile(cb_query, t, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            products.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, cb_products);
            tile_regs_release();
            products.push_back(1);
        }
        key.pop_front(global_tiles);
        query.pop_front(global_tiles);
    }

    {
        // 2. the chain's ttnn.sum over W: the accurate fp32 SFPU fold (the helper waits on the scaler, pops the input)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_products,
            cb_scaler,
            cb_sum,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Accurate>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, global_tiles, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});
        scaler.pop_front(1);
    }

    {
        // 3. the scalar chain on the sum tile: gate = sum * scale; |gate|, clamp, sqrt; sign(gate); product; sigmoid
        sum.wait_front(1);
        scale.wait_front(1);
        reconfig_data_format(cb_sum, cb_scale);
        pack_reconfig_data_format(cb_gate);
        tile_regs_acquire();
        copy_init(cb_sum);
        copy_tile(cb_sum, 0, 0);
        reconfig_data_format_srca(cb_sum, cb_scale);
        copy_init(cb_scale);
        copy_tile(cb_scale, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // gate (fp32)
        tile_regs_commit();
        gate.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_gate);
        if constexpr (debug) {
            DataflowBuffer debug_sum(cb_debug_sum);
            debug_sum.reserve_back(1);
            pack_reconfig_data_format(cb_debug_sum);
            pack_tile(0, cb_debug_sum);
            debug_sum.push_back(1);
            pack_reconfig_data_format(cb_gate);
        }
        tile_regs_release();
        gate.push_back(1);
        sum.pop_front(1);
        scale.pop_front(1);

        gate.wait_front(1);
        reconfig_data_format_srca(cb_scale, cb_gate);
        pack_reconfig_data_format(cb_coefficient);
        tile_regs_acquire();
        copy_init(cb_gate);
        copy_tile(cb_gate, 0, 0);  // magnitude chain on dst 0
        abs_tile_init();
        abs_tile(0);
        clamp_tile_init();
        clamp_tile(0, clamp_min_bits, clamp_max_bits);
        sqrt_tile_init();
        sqrt_tile<false>(0);
        copy_tile(cb_gate, 0, 1);  // direction on dst 1
        sign_tile_init();
        sign_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // transformed
        sigmoid_tile_init<0u>();
        sigmoid_tile<VectorMode::RC, 0u>(0);  // coefficient
        tile_regs_commit();
        coefficient.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_coefficient);
        if constexpr (debug) {
            DataflowBuffer debug_coefficient(cb_debug_coefficient);
            debug_coefficient.reserve_back(1);
            pack_reconfig_data_format(cb_debug_coefficient);
            pack_tile(0, cb_debug_coefficient);
            debug_coefficient.push_back(1);
            pack_reconfig_data_format(cb_coefficient);
        }
        tile_regs_release();
        coefficient.push_back(1);
        gate.pop_front(1);
    }

    {
        // 4. gated = value branches * coefficient (column broadcast), bf16 out: binary_ng's SFPU col-bcast kernel
        coefficient.wait_front(1);
        reconfig_data_format_srca(cb_gate, cb_coefficient);
        pack_reconfig_data_format(cb_broadcast);
        unary_bcast_init<BroadcastType::COL>(cb_coefficient);
        tile_regs_acquire();
        unary_bcast<BroadcastType::COL>(cb_coefficient, 0, 0);
        tile_regs_commit();
        broadcast.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_broadcast);
        tile_regs_release();
        broadcast.push_back(1);
        coefficient.pop_front(1);

        broadcast.wait_front(1);
        value.wait_front(local_tiles);
        pack_reconfig_data_format(cb_out);
        for (uint32_t t = 0; t < local_tiles; ++t) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_broadcast, cb_value);
            copy_init(cb_value);
            copy_tile(cb_value, t, 0);
            reconfig_data_format_srca(cb_value, cb_broadcast);
            copy_init(cb_broadcast);
            copy_tile(cb_broadcast, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            out.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            out.push_back(1);
        }
        value.pop_front(local_tiles);
        broadcast.pop_front(1);
    }
}
