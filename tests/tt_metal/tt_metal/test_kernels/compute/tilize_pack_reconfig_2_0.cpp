// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/tilize.h"
#include "api/compute/experimental/2_0/reconfig_data_format.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) counterpart of tilize_pack_reconfig.cpp: tilize_init arms the packer for a scratch output
// operand of a different L1 format (never written), pack_reconfig_data_format switches the packer to the
// real output, then tilize_block packs into c_16. Single tile per block.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr std::uint32_t block = 1;

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;
    // Scratch output of a different format and tile size (block float), same geometry; no L1 is written to it.
    using ScratchOutOp = experimental::LLKOperand<DataFormat::Bfp8_b, out_desc.shape>;

    compute_kernel_hw_startup(InOp(in_cb.read_address()), ScratchOutOp(out_cb.write_address()));
    experimental::tilize_init(InOp(in_cb.read_address()), block, ScratchOutOp(out_cb.write_address()));
    experimental::pack_reconfig_data_format(OutOp(out_cb.write_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb0.wait_front(block);
        cb16.reserve_back(block);

        experimental::tilize_block(InOp(in_cb.read_address()), block, OutOp(out_cb.write_address()));

        cb0.pop_front(block);
        cb16.push_back(block);
    }

    experimental::tilize_uninit(InOp(in_cb.read_address()), OutOp(out_cb.write_address()));
}
