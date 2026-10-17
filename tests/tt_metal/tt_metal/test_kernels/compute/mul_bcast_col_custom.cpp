// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blocked bcast-col MUL with SrcB reuse (indexer_score gate-mul), driven through the Compute API.
//
// Same block structure as sub_bcast_col_custom.cpp: a block is an RT_DIM x CT_DIM tile grid filling
// one acquired dest section; per row, ONE held srcB tile is column-broadcast and multiplied into that
// row's CT_DIM srcA column tiles, each product MAC'd onto its own (acquire-zeroed) dest slot.
//
// The output buffer may carry a larger tile than the inputs (full 32x32 out for 16x32 inputs): the
// packer then reads the whole 64-row dest slot, so rows the tiny-tile math must not touch are visible
// to the host as the lower half of each output tile.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/indexer_mul_custom.h"
#include "api/dataflow/dataflow_buffer.h"

#ifndef CT_DIM
#define CT_DIM 1
#endif

// Row tiles per block: the number of mul_tiles_bcast_cols_custom calls sharing one dest section.
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

    compute_kernel_hw_startup(icb0, icb1, ocb);
    mul_bcast_cols_init_short_custom(icb0, icb1);

    // One bcast tile per row of a block, held for the whole run and never popped per block, so
    // tile_index_b below indexes straight off the read pointer.
    dfb1.wait_front(rt_dim);

    for (std::uint32_t block = 0; block < num_blocks; block++) {
        dfb0.wait_front(tiles_per_block);
        dfb_out.reserve_back(tiles_per_block);

        tile_regs_acquire();
        for (std::uint32_t row = 0; row < rt_dim; row++) {
            // srcA tile index and dest base are the same offset: this row's start in the block grid.
            const std::uint32_t tile_base = row * ct_dim;
            mul_tiles_bcast_cols_custom(icb0, icb1, tile_base /*itile0*/, row /*itile1*/, tile_base /*idst*/, ct_dim);
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
