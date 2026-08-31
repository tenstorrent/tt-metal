// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/pack_untilize.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) pack-untilize with block_ct_dim = 4 (> 1). Reads 4 tiled input tiles in ONE window at
// in.l1_address + c * tile_stride_words(InFormat, InShape) and packs one row-major output row 4 tiles wide.
// With a block-float (Bfp8_b) INPUT this exercises the per-tile input stride for tile c > 0 -- the case where
// the old SCALE_DATUM_SIZE stride (68 vs 64 words; exponent bytes omitted) misreads tiles. Output must be
// bit-identical to pack_untilize_block4_legacy.cpp (differential; pure layout movement, byte-exact).
void kernel_main() {
    constexpr std::uint32_t BLOCK = 4;

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(InOp(in_cb.read_address()), OutOp(out_cb.write_address()));
    experimental::pack_untilize_init<BLOCK, BLOCK>(InOp(in_cb.read_address()), OutOp(out_cb.write_address()));

    cb0.wait_front(BLOCK);
    cb16.reserve_back(BLOCK);
    experimental::pack_untilize_block<BLOCK, BLOCK>(
        InOp(in_cb.read_address()), 1 /*block_rt_dim*/, OutOp(out_cb.write_address()));
    cb0.pop_front(BLOCK);
    cb16.push_back(BLOCK);

    experimental::pack_untilize_uninit(OutOp(out_cb.write_address()));
}
