// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/tilize.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) tilize kernel, classic circular buffers. The ops take LLKOperand (data format + tile
// geometry as NTTPs, L1 address the only runtime state). tilize_block owns the block loop and derives each
// tile's output slot from the compile-time output geometry (SCALE_DATUM_SIZE), so tile t lands in slot t.
// Output must be bit-identical to the legacy kernel tilize_legacy.cpp. This run uses block == 1 per CB slot.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(InOp(in_cb.read_address()), OutOp(out_cb.write_address()));
    experimental::tilize_init(InOp(in_cb.read_address()), 1 /*block*/, OutOp(out_cb.write_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        experimental::tilize_block(InOp(in_cb.read_address()), 1 /*block*/, OutOp(out_cb.write_address()));

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    experimental::tilize_uninit(InOp(in_cb.read_address()), OutOp(out_cb.write_address()));
}
