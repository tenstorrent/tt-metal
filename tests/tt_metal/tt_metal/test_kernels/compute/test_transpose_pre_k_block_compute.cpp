// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated test compute kernel for TransposePreKBlock
// (ttnn/cpp/ttnn/kernel_lib/transpose_block_helpers.hpp), used as the
// PreKBlockFn slot of matmul_block.
//
// CB layout matches production fused-bias kernel:
//   c_0  = cb_in0_transposed  — destination for the transposed tiles
//   c_3  = cb_in0             — ORIGINAL A tiles read by the host dataflow kernel
//                               (reader_matmul_blocked). This CB is passed as
//                               in0_transpose_cb_id into the functor, which
//                               reads, transposes WH, packs into c_0.
//   c_1  = cb_in1             — B input
//   c_16 = cb_out             — output
//   c_24 = cb_intermed0       — partials
//
// Reader pushes the "original" A into c_3 (not c_0). TransposePreKBlock drains
// c_3 → c_0. matmul_block then consumes c_0 as its in0_cb.
//
// Compile-time args:
//   [0]  in0_block_w
//   [1]  in0_num_subblocks
//   [2]  in0_block_num_tiles
//   [3]  in0_subblock_num_tiles
//   [4]  in1_num_subblocks
//   [5]  in1_block_num_tiles
//   [6]  in1_per_core_w
//   [7]  num_blocks
//   [8]  out_subblock_h
//   [9]  out_subblock_w
//   [10] out_subblock_num_tiles
//   [11] batch

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/transpose_block_helpers.hpp"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);
    constexpr uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;            // matmul_block's in0 (transposed)
    constexpr uint32_t in0_transpose_cb = tt::CBIndex::c_3;  // original A (reader target)
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t interm_cb = tt::CBIndex::c_24;

    // TransposePreKBlock functor: transposes in0_transpose_cb → in0_cb
    // at the start of every K-block iteration before matmul_block waits on in0.
    using XposeFn = compute_kernel_lib::TransposePreKBlock<
        in0_block_num_tiles,
        /*in0_transpose_cb_id=*/in0_transpose_cb,
        /*in0_cb_id=*/in0_cb,
        /*in1_cb_id=*/in1_cb,
        /*in1_transpose_tile=*/false,
        out_subblock_w,
        out_subblock_h,
        in0_block_w,
        /*mm_partials_cb_id=*/interm_cb>;

    mm_block_init(in0_cb, in1_cb, interm_cb, false, out_subblock_w, out_subblock_h, in0_block_w);

    compute_kernel_lib::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        /*pack_last_to_interm=*/false,
        /*pack_relu=*/false,
        /*row_major_output=*/false,
        compute_kernel_lib::NoPostCompute,
        XposeFn>(
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
        compute_kernel_lib::NoPostCompute{},
        XposeFn{});
}
