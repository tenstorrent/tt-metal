// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SDPA blocked bcast-col SUB with SrcB reuse, driven through the Compute API.
//
// A block is an RT_DIM x CT_DIM tile grid filling one acquired dest section. Per row of the block,
// ONE held srcB tile is column-broadcast and subtracted from that row's CT_DIM srcA column tiles,
// each difference landing in its own dest slot. That reuse is what distinguishes this op from
// calling the stock one-tile broadcast CT_DIM times.
//
// Mirrors the call sequence in sub_exp_block_bcast_cols() in
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp, including
// its per-row advance of the srcA tile index, the srcB tile index and the dest base within a single
// tile_regs_acquire().

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "api/dataflow/dataflow_buffer.h"

#ifndef CT_DIM
#define CT_DIM 1
#endif

// Row tiles per block: the number of sub_tiles_bcast_cols_custom calls sharing one dest section.
#ifndef RT_DIM
#define RT_DIM 1
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 1
#endif

void kernel_main() {
    constexpr std::uint32_t ct_dim = CT_DIM;
    constexpr std::uint32_t rt_dim = RT_DIM;
    constexpr std::uint32_t num_blocks = NUM_BLOCKS;
    constexpr std::uint32_t tiles_per_block = rt_dim * ct_dim;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr std::uint32_t icb0 = dfb::in0;
    constexpr std::uint32_t icb1 = dfb::in1;
    constexpr std::uint32_t ocb = dfb::out;

    binary_op_init_common(icb0, icb1, ocb);
    sub_bcast_cols_init_short_custom(icb0, icb1, ct_dim);

    // One bcast tile per row of a block, held for the whole run and never popped per block, so
    // tile_index_b below indexes straight off the read pointer.
    dfb1.wait_front(rt_dim);

    for (std::uint32_t block = 0; block < num_blocks; block++) {
        // A whole block must be resident: the op unpacks a row's ct_dim srcA tiles in one call.
        dfb0.wait_front(tiles_per_block);
        dfb_out.reserve_back(tiles_per_block);

        tile_regs_acquire();
        // tile_index_a is relative to the buffer read pointer, which pop_front advances per block,
        // so within a block it is just the row's offset into the grid. Row `row` writes dest slots
        // [row * ct_dim, (row + 1) * ct_dim).
        for (std::uint32_t row = 0; row < rt_dim; row++) {
            // srcA tile index and dest base are the same offset: this row's start in the block grid.
            const std::uint32_t tile_base = row * ct_dim;
            sub_tiles_bcast_cols_custom(icb0, icb1, tile_base /*itile0*/, row /*itile1*/, tile_base /*idst*/, ct_dim);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (std::uint32_t i = 0; i < tiles_per_block; i++) {
            pack_tile(i, ocb);
        }
        tile_regs_release();

        dfb_out.push_back(tiles_per_block);
        dfb0.pop_front(tiles_per_block);
    }

    dfb1.pop_front(rt_dim);
}
