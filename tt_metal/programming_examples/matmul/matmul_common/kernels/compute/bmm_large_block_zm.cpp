// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp"

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t num_k_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24;

    auto cfg = compute_kernel_lib::MatmulConfig::block(
        cb_in0, cb_in1, cb_out, out_subblock_w, out_subblock_h, in0_block_w, false, cb_interm);

    mm_block_init(cb_in0, cb_in1, cb_interm, false, out_subblock_w, out_subblock_h, in0_block_w);

    for (uint32_t b = 0; b < batch; b++) {
        compute_kernel_lib::matmul_blocks_absolute<compute_kernel_lib::BLOCK>(
            cfg, in0_num_subblocks, in1_num_subblocks, num_k_blocks);
    }
}
