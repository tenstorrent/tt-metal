// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) copy_block kernel: process the input in blocks of BLOCK_TILES; per block, copy_block a run
// of consecutive tiles c_0 -> DST, then pack each DST slot -> c_16. IDENTICAL to copy_block_legacy.cpp
// except the copy uses experimental::copy_block[_init] built from an LLKOperand (no CB id). pack_tile and
// compute_kernel_hw_startup stay the legacy CB-id API so the differential isolates copy_block. Output must be
// bit-for-bit identical to the legacy kernel.
constexpr std::uint32_t BLOCK_TILES = 4;  // fits DST for both fp32 and non-fp32 dest

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    experimental::copy_init(InOp(in_cb.read_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; b += BLOCK_TILES) {
        std::uint32_t ntiles = (per_core_tile_cnt - b) < BLOCK_TILES ? (per_core_tile_cnt - b) : BLOCK_TILES;

        tile_regs_acquire();
        cb0.wait_front(ntiles);
        cb16.reserve_back(ntiles);

        experimental::copy_block(InOp(in_cb.read_address()), 0, 0, ntiles);

        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t t = 0; t < ntiles; ++t) {
            pack_tile(t, tt::CBIndex::c_16);
        }

        cb0.pop_front(ntiles);
        cb16.push_back(ntiles);
        tile_regs_release();
    }
}
