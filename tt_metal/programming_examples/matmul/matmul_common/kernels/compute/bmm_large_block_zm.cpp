// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);        // K-dimension block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // sub-blocks along M dimension
    // args 2,3 are derived: in0_block_num_tiles, in0_subblock_num_tiles
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);  // sub-blocks along N dimension
    // args 5,6 are derived: in1_block_num_tiles, in1_per_core_w
    uint32_t num_k_blocks = get_compile_time_arg_val(7);    // blocks along K dimension
    uint32_t out_subblock_h = get_compile_time_arg_val(8);  // output sub-block height in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9);  // output sub-block width in tiles
    // arg 10 is derived: out_subblock_num_tiles
    uint32_t batch = get_compile_time_arg_val(11);  // batch dim

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24;

    compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
        in0_block_w, in0_num_subblocks, in1_num_subblocks, num_k_blocks, out_subblock_h, out_subblock_w, batch);
}
