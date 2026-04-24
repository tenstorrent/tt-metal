// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Parameterized test compute kernel for matmul_block helper
// (ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp).
//
// Exercises the template parameter pack: transpose, packer_l1_acc,
// pack_last_to_interm, pack_relu, OutputLayout, and PostComputeFn.
// PreKBlockFn coverage lives in the dedicated transpose-PreKBlock test.
//
// Defines controlling the helper template parameter pack:
//   HELPER_TRANSPOSE          — transpose=true
//   HELPER_PACKER_L1_ACC      — packer_l1_acc=true
//   HELPER_PACK_LAST_INTERM   — pack_last_to_interm=true  (writer reads c_24)
//   HELPER_PACK_RELU          — pack_relu=true
//   HELPER_ROW_MAJOR_OUTPUT   — layout=OutputLayout::RowMajor
//   HELPER_POST_COMPUTE_RELU  — PostComputeFn applies relu via SFPU
//
// CB layout:
//   c_0   (in0)       — Matrix A input
//   c_1   (in1)       — Matrix B input
//   c_16  (out)       — Final output  (used when pack_last_to_interm=false)
//   c_24  (interm)    — Intermediate / reload / L1-ACC FIFO. When
//                       pack_last_to_interm=true the writer reads c_24 here
//                       too (host swaps writer target accordingly).
//
// Compile-time args (match existing reader_matmul_blocked + writer_unswizzle
// conventions so the same host dataflow kernels can be reused):
//   [0]  in0_block_w
//   [1]  in0_num_subblocks
//   [2]  in0_block_num_tiles
//   [3]  in0_subblock_num_tiles
//   [4]  in1_num_subblocks
//   [5]  in1_block_num_tiles
//   [6]  in1_per_core_w
//   [7]  num_blocks (K-dim)
//   [8]  out_subblock_h
//   [9]  out_subblock_w
//   [10] out_subblock_num_tiles
//   [11] batch

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

#ifdef HELPER_POST_COMPUTE_RELU
#include "api/compute/eltwise_unary/relu.h"
#endif

namespace {

#ifdef HELPER_POST_COMPUTE_RELU
// Nontrivial PostComputeFn: applies relu per-tile via SFPU on the last K-block
// output before packing. Host-side golden applies ReLU for parity.
struct ReluPostCompute {
    ALWI void operator()(uint32_t num_tiles) const {
        relu_tile_init();
        for (uint32_t i = 0; i < num_tiles; i++) {
            relu_tile(i);
        }
    }
};
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);
    constexpr uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t interm_cb = tt::CBIndex::c_24;

#ifdef HELPER_TRANSPOSE
    constexpr bool transpose = true;
#else
    constexpr bool transpose = false;
#endif

#ifdef HELPER_PACKER_L1_ACC
    constexpr bool packer_l1_acc = true;
#else
    constexpr bool packer_l1_acc = false;
#endif

#ifdef HELPER_PACK_LAST_INTERM
    constexpr bool pack_last_to_interm = true;
#else
    constexpr bool pack_last_to_interm = false;
#endif

#ifdef HELPER_PACK_RELU
    constexpr bool pack_relu = true;
#else
    constexpr bool pack_relu = false;
#endif

#ifdef HELPER_ROW_MAJOR_OUTPUT
    constexpr compute_kernel_lib::OutputLayout output_layout = compute_kernel_lib::OutputLayout::RowMajor;
#else
    constexpr compute_kernel_lib::OutputLayout output_layout = compute_kernel_lib::OutputLayout::SubblockMajor;
#endif

    mm_block_init(in0_cb, in1_cb, interm_cb, transpose, out_subblock_w, out_subblock_h, in0_block_w);

#ifdef HELPER_POST_COMPUTE_RELU
    compute_kernel_lib::
        matmul_block<transpose, packer_l1_acc, pack_last_to_interm, pack_relu, output_layout, ReluPostCompute>(
            in0_cb,
            in1_cb,
            out_cb,
            interm_cb,
            in0_block_w,
            in0_num_subblocks,
            in1_num_subblocks,
            num_k_blocks,
            out_subblock_h,
            out_subblock_w,
            batch,
            ReluPostCompute{});
#else
    compute_kernel_lib::matmul_block<transpose, packer_l1_acc, pack_last_to_interm, pack_relu, output_layout>(
        in0_cb,
        in1_cb,
        out_cb,
        interm_cb,
        in0_block_w,
        in0_num_subblocks,
        in1_num_subblocks,
        num_k_blocks,
        out_subblock_h,
        out_subblock_w,
        batch);
#endif
}
