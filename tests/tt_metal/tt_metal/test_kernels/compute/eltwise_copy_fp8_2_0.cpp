// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Id-free (2.0) datacopy verification: the ops take an LLKOperand (data format + tile geometry as NTTPs,
// absolute L1 address as the only runtime state) -- NO CB id on the op surface. The register format is
// derived on-device from the L1 format. Legacy init is kept so this run isolates the id-free OP path +
// address seam + infer fn. Output must be bit-identical to the legacy path (eltwise_copy_fp8.cpp).
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Compile-time CB accessors -> folded descriptors; the operand bundles descriptor + runtime L1 address.
    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(InOp(in_cb.read_address()), OutOp(out_cb.write_address()));
    // fp8 input: the 2.0 experimental::copy_init static_asserts against fp8 (it does not wire the Src
    // zero-substitution flag), so fp8 datacopy MUST keep the legacy CB-id copy_tile_init here.
    copy_tile_init(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        cb0.wait_front(1);
        cb16.reserve_back(1);
        experimental::copy_tile(InOp(in_cb.read_address()), 0);

        tile_regs_commit();
        tile_regs_wait();

        experimental::pack_tile(OutOp(out_cb.write_address()), 0);
        cb0.pop_front(1);
        cb16.push_back(1);

        tile_regs_release();
    }
}
