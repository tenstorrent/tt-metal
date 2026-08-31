// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"  // legacy copy_tile OP (the copy-into-DST op stays CB-id)
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"  // id-free copy_init
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) pack_block kernel: process the input in blocks of 4 tiles. IDENTICAL to pack_block_legacy.cpp
// except the block pack uses experimental::pack_block, built from an output LLKOperand (data format + tile
// geometry as NTTPs, absolute L1 address the only runtime state). compute_kernel_hw_startup and the input
// copy_tile stay the legacy CB-id API so the differential isolates pack_block. pack_block loops over the 2.0
// pack_tile, deriving each tile's output address from the compile-time output tile stride
// (out.l1_address + i * tile_stride_words) -- the same consecutive L1 tiles the legacy in-order pack loop
// writes. Compile-time arg 0 = total tile count (assumed a multiple of 4). Output must be bit-for-bit identical
// to the legacy kernel.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr std::uint32_t block = 4;

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Compile-time CB accessors -> folded descriptors; the operand bundles descriptor + runtime L1 address.
    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    // Id-free copy-into-DST init from the input CB (c_0); the copy_tile OP below stays the legacy CB-id call.
    experimental::copy_init(InOp(in_cb.read_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; b += block) {
        tile_regs_acquire();
        cb0.wait_front(block);
        cb16.reserve_back(block);

        for (std::uint32_t i = 0; i < block; ++i) {
            copy_tile(tt::CBIndex::c_0, i, i);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack DST[0..block-1] to the block base (reserved-region tile 0); pack_block strides by tile_stride_words.
        experimental::pack_block(OutOp(out_cb.write_address()), 0 /*ifrom_dst*/, block);

        cb0.pop_front(block);
        cb16.push_back(block);
        tile_regs_release();
    }
}
