// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The down worker of the GR read with the rsqrt applied LAST (ttnn/fused/gr_recip_last), one output column tile:
//   1. matmul: the multicast u row (k_tiles tiles = one stream of branch_k_tiles tiles per branch) times the core's
//      weight column, K tile k accumulated into dest tile k / branch_k_tiles, so the branches' partials P_b stay
//      apart (fp32 dest, no spill).
//   2. scale: P_b * recip_b on the FPU (mul_tiles_bcast_cols, recip_b column 0 broadcast), packed fp32.
//   3. sum over the branches in branch order with the zero tile (lowrank_compute.cpp's fold), packed fp32: the
//      device's partial tile for the line gather, on the chain's partial page.
// CBs (compile-time): 3 in0 u row (bf16, k_tiles), 4 in1 weight column (bf16, k_tiles, pushed block_tiles at a time),
// 5 interm (fp32, branches), 6 recip (fp32, branches), 7 products (fp32, branches), 8 zero (fp32, 1), 9 out (fp32, 1).
// Compile-time args: 0 k_tiles, 1 block_tiles (weight tiles per push, divides k_tiles), 2 branch_k_tiles (K tiles per
// branch), 3-9 the CBs, 10 branches.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t branch_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t c_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t c_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t c_interm = get_compile_time_arg_val(5);
    constexpr uint32_t c_recip = get_compile_time_arg_val(6);
    constexpr uint32_t c_prod = get_compile_time_arg_val(7);
    constexpr uint32_t c_zero = get_compile_time_arg_val(8);
    constexpr uint32_t c_out = get_compile_time_arg_val(9);
    constexpr uint32_t branches = get_compile_time_arg_val(10);
    static_assert(
        k_tiles % block_tiles == 0 && k_tiles == branches * branch_k_tiles && branches <= 8,
        "K tiles split into one stream of branch_k_tiles per branch, pushed block_tiles at a time");

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_in0, c_in1, c_interm);
    matmul_block_init(c_in0, c_in1, 0, 1, 1, 1);
    DataflowBuffer in0(c_in0);
    DataflowBuffer in1(c_in1);
    DataflowBuffer interm(c_interm);
    DataflowBuffer recip(c_recip);
    DataflowBuffer prod(c_prod);
    DataflowBuffer zero(c_zero);
    DataflowBuffer out(c_out);

    {
        in0.wait_front(k_tiles);
        tile_regs_acquire();
        for (uint32_t k = 0; k < k_tiles; k += block_tiles) {
            in1.wait_front(block_tiles);
            for (uint32_t kk = 0; kk < block_tiles; ++kk) {
                matmul_block(c_in0, c_in1, k + kk, kk, (k + kk) / branch_k_tiles, 0, 1, 1, 1);
            }
            in1.pop_front(block_tiles);
        }
        tile_regs_commit();
        interm.reserve_back(branches);
        tile_regs_wait();
        pack_reconfig_data_format(c_interm);
        for (uint32_t b = 0; b < branches; ++b) {
            pack_tile(b, c_interm);
        }
        tile_regs_release();
        interm.push_back(branches);
        in0.pop_front(k_tiles);
    }
    {
        recip.wait_front(branches);
        interm.wait_front(branches);
        reconfig_data_format(c_interm, c_recip);
        pack_reconfig_data_format(c_prod);
        mul_bcast_cols_init(c_interm, c_recip);
        tile_regs_acquire();
        for (uint32_t b = 0; b < branches; ++b) {
            mul_tiles_bcast_cols(c_interm, c_recip, b, b, b);
        }
        tile_regs_commit();
        prod.reserve_back(branches);
        tile_regs_wait();
        for (uint32_t b = 0; b < branches; ++b) {
            pack_tile(b, c_prod);
        }
        tile_regs_release();
        prod.push_back(branches);
        interm.pop_front(branches);
        recip.pop_front(branches);
    }
    {
        prod.wait_front(branches);
        zero.wait_front(1);
        add_init(c_prod, c_zero, true);
        reconfig_data_format(c_prod, c_zero);
        pack_reconfig_data_format(c_out);
        tile_regs_acquire();
        for (uint32_t b = 0; b < branches; ++b) {
            add_tiles(c_prod, c_zero, b, 0, 0);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_out);
        tile_regs_release();
        out.push_back(1);
        prod.pop_front(branches);
        zero.pop_front(1);
    }
}
