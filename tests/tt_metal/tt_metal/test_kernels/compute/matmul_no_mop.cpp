// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blocked matmul driven through the experimental MOP-less Compute API (issue #52329).
//
// out[RT_DIM x CT_DIM] = in0[RT_DIM x KT_DIM] * in1[KT_DIM x CT_DIM], accumulated in DEST across the
// KT_DIM inner steps of one dest acquire. The no-MOP path issues REPLAY + MVMUL straight from the
// RISC core instead of running MOP BANK0, so the result must match the MOP matmul exactly.
//
// Structure mirrors matmul_block.cpp (the DFB-based MOP matmul kernel): the LLK does not walk kt_dim,
// so the kernel loops it. in0 is [RT_DIM x KT_DIM] row-major and in1 is [KT_DIM x CT_DIM] row-major,
// so step k reads in0 tile k (the LLK strides down the rows by kt_dim) and in1 tile k * CT_DIM.

#include <cstdint>
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr std::uint32_t ct_dim = CT_DIM;
    constexpr std::uint32_t rt_dim = RT_DIM;
    constexpr std::uint32_t kt_dim = KT_DIM;

    constexpr std::uint32_t in0_num_tiles = rt_dim * kt_dim;
    constexpr std::uint32_t in1_num_tiles = kt_dim * ct_dim;
    constexpr std::uint32_t out_num_tiles = rt_dim * ct_dim;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr std::uint32_t icb0 = dfb::in0;
    constexpr std::uint32_t icb1 = dfb::in1;
    constexpr std::uint32_t ocb = dfb::out;

    // Matmul maps in0 -> SrcB and in1 -> SrcA, the reverse of other ops, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(icb0, icb1, ocb);
    mm_no_mop_init_short(icb0, icb1, false /*transpose*/, ct_dim, rt_dim, kt_dim);

    // The whole block of both operands must be resident: one matmul_block_no_mop call unpacks a
    // rt_dim x 1 slice of in0 and a 1 x ct_dim slice of in1, indexed off the read pointer.
    dfb0.wait_front(in0_num_tiles);
    dfb1.wait_front(in1_num_tiles);
    dfb_out.reserve_back(out_num_tiles);

    tile_regs_acquire();
    for (std::uint32_t k = 0; k < kt_dim; k++) {
        matmul_block_no_mop(
            icb0,
            icb1,
            k /*in0_tile_index*/,
            k * ct_dim /*in1_tile_index*/,
            0 /*idst*/,
            false /*transpose*/,
            ct_dim,
            rt_dim,
            kt_dim);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (std::uint32_t i = 0; i < out_num_tiles; i++) {
        pack_tile(i, ocb);
    }
    tile_regs_release();

    dfb_out.push_back(out_num_tiles);
    dfb0.pop_front(in0_num_tiles);
    dfb1.pop_front(in1_num_tiles);
}
