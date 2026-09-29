// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The down worker of the GR read with the rsqrt applied LAST (ttnn/fused/gr_recip_last), one output column tile:
//   1. matmul: the multicast u row (Kt tiles = B streams x Kb tiles each) times the core's weight column, K tile k
//      accumulated into dest tile k / Kb, so the B streams' partials P_b stay apart (fp32 dest, no spill: the
//      chain's DRAM-sharded spill/reload is a rounding this class does not reproduce).
//   2. scale: P_b * recip_b on the FPU (mul_tiles_bcast_cols: the chain's x*rsqrt instruction, here on the fp32
//      partial; recip_b is the norm core's rsqrt tile, column 0 broadcast), packed fp32.
//   3. sum over the B streams in stream order with the zero tile (lowrank_compute.cpp's fold), packed fp32: the
//      device's partial tile for the line gather, the same page the chain's partial takes.
// CBs (compile-time): 3 in0 u row (bf16, Kt), 4 in1 weight column (bf16, Kt, pushed blk at a time), 5 interm (fp32,
// B), 6 recip (fp32, B), 7 products (fp32, B), 8 zero (fp32, 1), 9 out (fp32, 1).
// Compile-time args: 0 Kt, 1 blk (weight tiles per push, divides Kt), 2 Kb (K tiles per stream), 3-9 the CBs, 10 B.

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
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t Kb = get_compile_time_arg_val(2);
    constexpr uint32_t c_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t c_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t c_interm = get_compile_time_arg_val(5);
    constexpr uint32_t c_recip = get_compile_time_arg_val(6);
    constexpr uint32_t c_prod = get_compile_time_arg_val(7);
    constexpr uint32_t c_zero = get_compile_time_arg_val(8);
    constexpr uint32_t c_out = get_compile_time_arg_val(9);
    constexpr uint32_t B = get_compile_time_arg_val(10);
    static_assert(Kt % blk == 0 && Kt == B * Kb && B <= 8, "K tiles split into B streams of Kb, pushed blk at a time");

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
        FUSED_ZONE("fz_gl_d_matmul");
        in0.wait_front(Kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < Kt; k += blk) {
            in1.wait_front(blk);
            for (uint32_t kk = 0; kk < blk; ++kk) {
                matmul_block(c_in0, c_in1, k + kk, kk, (k + kk) / Kb, 0, 1, 1, 1);
            }
            in1.pop_front(blk);
        }
        tile_regs_commit();
        interm.reserve_back(B);
        tile_regs_wait();
        pack_reconfig_data_format(c_interm);
        for (uint32_t b = 0; b < B; ++b) {
            pack_tile(b, c_interm);
        }
        tile_regs_release();
        interm.push_back(B);
        in0.pop_front(Kt);
    }
    {
        FUSED_ZONE("fz_gl_d_scale");
        recip.wait_front(B);
        interm.wait_front(B);
        reconfig_data_format(c_interm, c_recip);
        pack_reconfig_data_format(c_prod);
        mul_bcast_cols_init(c_interm, c_recip);
        tile_regs_acquire();
        for (uint32_t b = 0; b < B; ++b) {
            mul_tiles_bcast_cols(c_interm, c_recip, b, b, b);
        }
        tile_regs_commit();
        prod.reserve_back(B);
        tile_regs_wait();
        for (uint32_t b = 0; b < B; ++b) {
            pack_tile(b, c_prod);
        }
        tile_regs_release();
        prod.push_back(B);
        interm.pop_front(B);
        recip.pop_front(B);
    }
    {
        FUSED_ZONE("fz_gl_d_sum");
        prod.wait_front(B);
        zero.wait_front(1);
        add_init(c_prod, c_zero, true);
        reconfig_data_format(c_prod, c_zero);
        pack_reconfig_data_format(c_out);
        tile_regs_acquire();
        for (uint32_t b = 0; b < B; ++b) {
            add_tiles(c_prod, c_zero, b, 0, 0);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, c_out);
        tile_regs_release();
        out.push_back(1);
        prod.pop_front(B);
        zero.pop_front(1);
    }
}
