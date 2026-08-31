// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/pack_untilize.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) pack-untilize kernel, classic circular buffers. The ops take LLKOperand (data format + tile
// geometry as NTTPs, L1 address the only runtime state). pack_untilize_block owns the row/column loops and
// derives per-tile input/output addresses from the compile-time geometry (SCALE_DATUM_SIZE). Output must be
// bit-identical to the legacy kernel pack_untilize_legacy.cpp. This run uses block_ct_dim/full_ct_dim/
// block_rt_dim == 1 (one tile per CB slot).
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
    experimental::pack_untilize_init<1 /*block_ct_dim*/, 1 /*full_ct_dim*/>(
        InOp(in_cb.read_address()), OutOp(out_cb.write_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        experimental::pack_untilize_block<1 /*block_ct_dim*/, 1 /*full_ct_dim*/>(
            InOp(in_cb.read_address()), 1 /*block_rt_dim*/, OutOp(out_cb.write_address()));

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    experimental::pack_untilize_uninit(OutOp(out_cb.write_address()));
}
