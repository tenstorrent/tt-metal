// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"  // pack_tile (legacy CB-id API, kept identical to the legacy kernel)
#include "api/compute/experimental/2_0/transpose.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/dataflow/circular_buffer.h"

// Id-free (2.0) transpose kernel: per tile, unpack+transpose c_0 directly into DST (via
// experimental::transpose_init / experimental::transpose_tile), pack -> c_16. IDENTICAL to
// transpose_legacy.cpp except the transpose init/op calls take an LLKOperand instead of a CB id. pack_tile
// stays the legacy CB-id API so the differential isolates transpose_init/transpose_tile. Output must be
// bit-for-bit identical to the legacy kernel.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using SrcOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    experimental::transpose_init(SrcOp(in_cb.read_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        experimental::transpose_tile(SrcOp(in_cb.read_address()), 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
