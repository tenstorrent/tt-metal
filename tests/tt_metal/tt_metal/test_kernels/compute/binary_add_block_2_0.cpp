// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"  // compute_kernel_hw_startup + add_init + tile_regs_* handshake
#include "api/compute/tile_move_copy.h"  // legacy pack_tile
#include "api/compute/experimental/2_0/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) eltwise-binary ADD block kernel: process two inputs (c_0, c_1) in blocks of BLOCK_TILES; per
// block, add_block a run of consecutive tile pairs c_0[i]+c_1[i] -> DST[i], then pack each DST slot -> c_16.
// IDENTICAL to binary_add_block_legacy.cpp except the block add uses experimental::add_block[/add_init] built
// from LLKOperands (no CB id). compute_kernel_hw_startup / pack_tile stay the legacy CB-id API so the
// differential isolates add_block. Output must be bit-for-bit identical to the legacy kernel. num_tiles must be
// a multiple of BLOCK_TILES and the CBs at least BLOCK_TILES deep (a block keeps BLOCK_TILES tiles resident via
// wait_front, so a 1-deep CB deadlocks).
constexpr std::uint32_t BLOCK_TILES = 4;  // fits DST for both fp32 and non-fp32 dest

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in0_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in1_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto in0_desc = experimental::to_llk_mem_descriptor(in0_cb);
    constexpr auto in1_desc = experimental::to_llk_mem_descriptor(in1_cb);
    using AOp = experimental::LLKOperand<static_cast<DataFormat>(in0_desc.format), in0_desc.shape>;
    using BOp = experimental::LLKOperand<static_cast<DataFormat>(in1_desc.format), in1_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    experimental::add_init(AOp(in0_cb.read_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; b += BLOCK_TILES) {
        std::uint32_t ntiles = (per_core_tile_cnt - b) < BLOCK_TILES ? (per_core_tile_cnt - b) : BLOCK_TILES;

        tile_regs_acquire();
        cb0.wait_front(ntiles);
        cb1.wait_front(ntiles);
        cb16.reserve_back(ntiles);

        experimental::add_block(AOp(in0_cb.read_address()), BOp(in1_cb.read_address()), 0, 0, 0, ntiles);

        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t t = 0; t < ntiles; ++t) {
            pack_tile(t, tt::CBIndex::c_16);
        }

        cb0.pop_front(ntiles);
        cb1.pop_front(ntiles);
        cb16.push_back(ntiles);
        tile_regs_release();
    }
}
