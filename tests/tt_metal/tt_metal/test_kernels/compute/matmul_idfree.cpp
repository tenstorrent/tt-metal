// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"  // compute_kernel_hw_startup, SrcOrder, tile_regs_*
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/matmul.h"
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) matmul kernel, classic circular buffers. Mirrors the shipping legacy matmul.cpp structure
// exactly (same 7 compile-time args + loop nest) but drives it with the id-free ops: one LLKOperand per input
// (in0 -> SrcB, in1 -> SrcA), format+geometry as NTTPs, L1 addresses the only runtime state. Output must be
// bit-identical to matmul.cpp. Named "_idfree" (not "_2_0"): matmul_2_0 semantics differ elsewhere.
void kernel_main() {
    std::uint32_t block_tile_dim = get_compile_time_arg_val(0);
    std::uint32_t dst_tile_rows = get_compile_time_arg_val(1);
    std::uint32_t dst_tile_cols = get_compile_time_arg_val(2);
    std::uint32_t block_cnt = get_compile_time_arg_val(3);
    std::uint32_t in0_block_tile_cnt = get_compile_time_arg_val(4);
    std::uint32_t in1_block_tile_cnt = get_compile_time_arg_val(5);
    std::uint32_t out_block_tile_cnt = get_compile_time_arg_val(6);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in0_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in1_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto d0 = experimental::to_llk_mem_descriptor(in0_cb);
    constexpr auto d1 = experimental::to_llk_mem_descriptor(in1_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using In0Op = experimental::LLKOperand<static_cast<DataFormat>(d0.format), d0.shape>;
    using In1Op = experimental::LLKOperand<static_cast<DataFormat>(d1.format), d1.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        In0Op(in0_cb.read_address()), In1Op(in1_cb.read_address()), OutOp(out_cb.write_address()));
    experimental::matmul_init(In0Op(in0_cb.read_address()), In1Op(in1_cb.read_address()));

    tile_regs_acquire();
    for (std::uint32_t b = 0; b < block_cnt; ++b) {
        cb0.wait_front(in0_block_tile_cnt);
        cb1.wait_front(in1_block_tile_cnt);
        int dst_tile_index = 0;
        int in0_block_tile_index = 0;
        for (std::uint32_t r = 0; r < dst_tile_rows; ++r) {
            for (std::uint32_t c = 0; c < dst_tile_cols; ++c) {
                int in1_block_tile_index = 0;
                for (std::uint32_t i = 0; i < block_tile_dim; ++i) {
                    experimental::matmul_tiles(
                        In0Op(in0_cb.read_address()),
                        In1Op(in1_cb.read_address()),
                        in0_block_tile_index + i,
                        in1_block_tile_index + c,
                        dst_tile_index);
                    in1_block_tile_index += dst_tile_cols;
                }
                dst_tile_index++;
            }
            in0_block_tile_index += block_tile_dim;
        }
        cb0.pop_front(in0_block_tile_cnt);
        cb1.pop_front(in1_block_tile_cnt);
    }

    tile_regs_commit();
    tile_regs_wait();

    cb16.reserve_back(out_block_tile_cnt);
    for (std::uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        experimental::pack_tile(OutOp(out_cb.write_address()), i, i);
    }
    cb16.push_back(out_block_tile_cnt);

    tile_regs_release();
}
