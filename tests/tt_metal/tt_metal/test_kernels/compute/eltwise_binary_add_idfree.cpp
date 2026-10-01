// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"  // compute_kernel_hw_startup + tile_regs_* handshake
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/eltwise_binary.h"
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) eltwise-binary ADD kernel, classic circular buffers. The ops take one LLKOperand per input
// (format + geometry as NTTPs, L1 address the only runtime state); binary is format-free so only geometry +
// addresses flow through. Output must be bit-identical to the legacy kernel eltwise_binary_add_legacy.cpp.
// Named "_idfree" (not "_2_0") to avoid colliding with the pre-existing metal-2.0 eltwise_binary_2_0.cpp.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer c0(tt::CBIndex::c_0);
    CircularBuffer c1(tt::CBIndex::c_1);
    CircularBuffer c16(tt::CBIndex::c_16);

    constexpr auto in0_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in1_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in0_desc = experimental::to_llk_mem_descriptor(in0_cb);
    constexpr auto in1_desc = experimental::to_llk_mem_descriptor(in1_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using AOp = experimental::LLKOperand<static_cast<DataFormat>(in0_desc.format), in0_desc.shape>;
    using BOp = experimental::LLKOperand<static_cast<DataFormat>(in1_desc.format), in1_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(AOp(in0_cb.read_address()), BOp(in1_cb.read_address()), OutOp(out_cb.write_address()));
    experimental::add_init(AOp(in0_cb.read_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        c0.wait_front(1);
        c1.wait_front(1);
        c16.reserve_back(1);

        tile_regs_acquire();
        experimental::add_tiles(AOp(in0_cb.read_address()), BOp(in1_cb.read_address()), 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        experimental::pack_tile(OutOp(out_cb.write_address()), 0, 0);
        tile_regs_release();

        c0.pop_front(1);
        c1.pop_front(1);
        c16.push_back(1);
    }
}
