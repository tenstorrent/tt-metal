// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ttnn.rms_norm(x bf16, weight bf16, eps) under an fp32 dest, call for call: layernorm.cpp's RMSNORM + FUSE_GAMMA
// path (x*x on the FPU packed fp32, SUM reduce against the 1.0 scaler then the SFPU 1/W, + eps, rsqrt, x * rsqrt
// packed fp32, * gamma packed bf16) on the same CB formats.  Bitwise against the op (fused_qsa_llk_pins, 2026-09-14).
// CBs: c_in x (bf16, Wt), c_scaler (bf16, 1, never popped), c_eps (bf16, 1, never popped), c_gamma (bf16, Wt, never
// popped), c_xmm2 (fp32, Wt), c_ex2 (fp32, 1), c_ex2pe (fp32, 1), c_fusion (fp32, 2*block), c_out (bf16, Wt).
// The caller runs compute_kernel_hw_startup once; the kernel must be compiled with fp32_dest_acc_en.

#pragma once

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"

template <
    uint32_t Wt,
    uint32_t W,
    uint32_t c_in,
    uint32_t c_scaler,
    uint32_t c_eps,
    uint32_t c_gamma,
    uint32_t c_xmm2,
    uint32_t c_ex2,
    uint32_t c_ex2pe,
    uint32_t c_fusion,
    uint32_t c_out>
inline void rms_norm_rows(uint32_t rows) {
    namespace generic = norm::kernel_util::generic;
    namespace numeric = norm::kernel_util::compute::numeric;
    namespace policies = norm::kernel_util::compute::policies;
    constexpr uint32_t block_size = 4;  // layernorm_op_multi_core.cpp: fp32_dest_acc_en ? 4 : 8
    constexpr int dst0 = 0;

    DataflowBuffer in(c_in), scaler(c_scaler), eps(c_eps), gamma(c_gamma), xmm2(c_xmm2), ex2(c_ex2), ex2pe(c_ex2pe),
        fusion(c_fusion), out(c_out);
    eps.wait_front(1);
    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();

    for (uint32_t r = 0; r < rows; ++r) {
        reconfig_data_format(c_in, c_in);
        pack_reconfig_data_format(c_xmm2);
        mul_init(c_in, c_in);
        for (auto block : generic::blocks(Wt, block_size)) {
            in.wait_front(block.start() + block.full_block_size());
            tile_regs_acquire();
            for (auto i : block.local()) {
                const auto g = block.to_global(i);
                mul_tiles(c_in, c_in, g, g, i);
            }
            tile_regs_commit();
            xmm2.reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, c_xmm2);
            }
            tile_regs_release();
            xmm2.push_back(block.full_block_size());
        }
        reconfig_data_format(c_in, c_xmm2, c_in, c_scaler);

        numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, true, policies::FullBlockWithPopPolicy>(
            xmm2, scaler, ex2, W, Wt, block_size, 32);

        ex2.wait_front(1);
        reconfig_data_format(c_ex2, c_eps);
        tile_regs_acquire();
        add_init(c_ex2, c_eps);
        add_tiles(c_ex2, c_eps, 0, 0, dst0);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(dst0);
        tile_regs_commit();
        ex2.pop_front(1);
        ex2pe.reserve_back(1);
        pack_reconfig_data_format(c_ex2pe);
        tile_regs_wait();
        pack_tile(dst0, c_ex2pe);
        tile_regs_release();
        ex2pe.push_back(1);

        ex2pe.wait_front(1);
        for (auto block : generic::blocks(Wt, block_size)) {
            reconfig_data_format(c_in, c_ex2pe);
            pack_reconfig_data_format(c_fusion);
            fusion.reserve_back(block.full_block_size());
            reconfig_data_format_srca(c_fusion, c_in);
            tile_regs_acquire();
            mul_bcast_cols_init(c_in, c_ex2pe);
            for (auto i : block.local()) {
                mul_tiles_bcast_cols(c_in, c_ex2pe, block.to_global(i), 0, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, c_fusion);
            }
            tile_regs_release();
            fusion.push_back(block.full_block_size());
            reconfig_data_format_srca(c_in, c_fusion);

            pack_reconfig_data_format(c_out);
            reconfig_data_format_srcb(c_ex2pe, c_gamma);
            gamma.wait_front(block.start() + block.full_block_size());
            fusion.wait_front(block.full_block_size());
            tile_regs_acquire();
            mul_bcast_rows_init(c_fusion, c_gamma);
            for (auto i : block.local()) {
                mul_tiles_bcast_rows(c_fusion, c_gamma, i, block.to_global(i), i);
            }
            tile_regs_commit();
            fusion.pop_front(block.full_block_size());
            out.reserve_back(block.full_block_size());
            tile_regs_wait();
            for (auto i : block.local()) {
                pack_tile(i, c_out);
            }
            tile_regs_release();
            out.push_back(block.full_block_size());
        }
        ex2pe.pop_front(1);
        in.pop_front(total_buffer_size);
    }
}
