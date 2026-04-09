// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Parameterized test compute kernel for matmul_block and add_bias_bcast_rows helpers.
//
// Features controlled by JIT defines (set via ComputeConfig::defines):
//   PACKER_L1_ACC  — Use hardware L1 accumulation instead of software spill/reload
//   PACK_RELU      — Enable RELU on the pack phase
//   FUSE_BIAS      — Enable bias-add phase after matmul (uses add_bias_bcast_rows helper)
//
// CB layout:
//   c_0  (in0)    — Matrix A input
//   c_1  (in1)    — Matrix B input
//   c_2  (bias)   — Bias input (only when FUSE_BIAS)
//   c_16 (out)    — Final output
//   c_24 (interm) — Intermediate partials / L1 accumulation FIFO
//
// Compile-time args (same layout as existing matmul test kernels):
//   [0]  in0_block_w           — K-dimension block size in tiles
//   [1]  in0_num_subblocks     — sub-blocks along M dimension
//   [2]  in0_block_num_tiles   — (derived: out_subblock_h * in0_block_w * in0_num_subblocks)
//   [3]  in0_subblock_num_tiles— (derived: out_subblock_h * in0_block_w)
//   [4]  in1_num_subblocks     — sub-blocks along N dimension
//   [5]  in1_block_num_tiles   — (derived: out_subblock_w * in0_block_w * in1_num_subblocks)
//   [6]  in1_per_core_w        — (derived: out_subblock_w * in1_num_subblocks)
//   [7]  num_blocks            — K-dimension blocks
//   [8]  out_subblock_h        — output sub-block height in tiles
//   [9]  out_subblock_w        — output sub-block width in tiles
//   [10] out_subblock_num_tiles— (derived: out_subblock_h * out_subblock_w)
//   [11] batch

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#endif
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"

void kernel_main() {
    // ── Compile-time dimensions ──
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t batch = get_compile_time_arg_val(11);

    // ── CB indices ──
    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t bias_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t interm_cb = tt::CBIndex::c_24;

    // ── Feature flags from defines ──
    constexpr bool xpose = false;

#ifdef PACKER_L1_ACC
    constexpr bool l1_acc = true;
#else
    constexpr bool l1_acc = false;
#endif

    // ── Initialize matmul block engine (caller-managed, per design doc) ──
    mm_block_init(in0_cb, in1_cb, interm_cb, xpose, out_subblock_w, out_subblock_h, in0_block_w);

#ifdef FUSE_BIAS
    // ── Path: matmul → pack to interm → bias add → pack to out ──

    compute_kernel_lib::matmul_block<
        in0_cb,
        in1_cb,
        out_cb,
        interm_cb,
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        xpose,
        l1_acc,
        /*pack_last_to_interm=*/true>(in0_block_w, num_blocks, batch);

    // Caller-managed transition between matmul and bias phases
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Bias add: reads matmul output from interm_cb, adds row-broadcast bias, packs to out_cb.
    constexpr uint32_t bias_width_tiles = in1_per_core_w;
    compute_kernel_lib::add_bias_bcast_rows<interm_cb, bias_cb, out_cb>(
        in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, bias_width_tiles);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif

#else  // !FUSE_BIAS
    // ── Path: matmul → pack directly to out ──
    // PACK_RELU: use HwRelu (zero-cost packer hardware), otherwise NoPostCompute.
#ifdef PACK_RELU
    using ReluPostFn = compute_kernel_lib::matmul_block_config::HwRelu;
#else
    using ReluPostFn = compute_kernel_lib::matmul_block_config::NoPostCompute;
#endif

    compute_kernel_lib::matmul_block<
        in0_cb,
        in1_cb,
        out_cb,
        interm_cb,
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        xpose,
        l1_acc,
        /*pack_last_to_interm=*/false,
        ReluPostFn>(in0_block_w, num_blocks, batch, ReluPostFn{});

#endif  // FUSE_BIAS
}
