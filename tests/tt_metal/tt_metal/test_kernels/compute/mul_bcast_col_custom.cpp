// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blocked bcast-col MUL with SrcB reuse (indexer_score gate-mul), driven through the Compute API.
//
// Same block structure as sub_bcast_col_custom.cpp: a block is an RT_DIM x CT_DIM tile grid filling
// one acquired dest section; per row, ONE held srcB tile is column-broadcast and multiplied into that
// row's CT_DIM srcA column tiles, each product MAC'd onto its own (acquire-zeroed) dest slot. With
// ACCUMULATE_ROWS every row MACs onto the same CT_DIM slots instead, as indexer_score reduces heads.
//
// The output buffer may carry a larger tile than the inputs (full 32x32 out for 16x32 inputs): the
// packer then reads the whole 64-row dest slot, so rows the tiny-tile math must not touch are visible
// to the host as the lower half of each output tile.
//
// PRIME_FULL_TILES: before the blocks, one PRIME_RT_DIM x PRIME_CT_DIM block of full 32x32 tiles from
// in2/in3 runs through the same op in this same program. Every kernel starts with both Src banks
// zeroed, so without it a walk that reads Src rows a tiny tile never wrote would MAC zeros and stay
// invisible; after it, with at least two rows and two columns, every bank of SrcA and SrcB holds
// nonzero data in rows 32-63.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/indexer_mul_custom.h"
#include "api/compute/reconfig_data_format.h"
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

#ifndef ACCUMULATE_ROWS
#define ACCUMULATE_ROWS 0
#endif

#ifndef PRIME_FULL_TILES
#define PRIME_FULL_TILES 0
#endif

namespace {

// One block: rt_dim rows of ct_dim srcA tiles against the first rt_dim tiles held at the front of
// dfb_b, in one acquired dest section, packed as the next output tiles. Row `row` writes dest slots
// [row * ct_dim, (row + 1) * ct_dim), or slots [0, ct_dim) for every row when accumulating.
template <bool accumulate_rows>
void mul_bcast_col_block(
    DataflowBuffer& dfb_a,
    DataflowBuffer& dfb_out,
    std::uint32_t icb_a,
    std::uint32_t icb_b,
    std::uint32_t ocb,
    std::uint32_t rt_dim,
    std::uint32_t ct_dim) {
    const std::uint32_t tiles_per_block = rt_dim * ct_dim;
    const std::uint32_t out_tiles = accumulate_rows ? ct_dim : tiles_per_block;

    // A whole block must be resident: the op unpacks a row's ct_dim srcA tiles in one call.
    dfb_a.wait_front(tiles_per_block);
    dfb_out.reserve_back(out_tiles);

    tile_regs_acquire();
    for (std::uint32_t row = 0; row < rt_dim; row++) {
        // srcA tile index and dest base are the same offset: this row's start in the block grid.
        const std::uint32_t tile_base = row * ct_dim;
        mul_tiles_bcast_cols_custom(
            icb_a, icb_b, tile_base /*itile0*/, row /*itile1*/, accumulate_rows ? 0 : tile_base /*idst*/, ct_dim);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (std::uint32_t i = 0; i < out_tiles; i++) {
        pack_tile(i, ocb);
    }
    tile_regs_release();

    dfb_out.push_back(out_tiles);
    dfb_a.pop_front(tiles_per_block);
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t ct_dim = CT_DIM;
    constexpr std::uint32_t rt_dim = RT_DIM;
    constexpr std::uint32_t num_blocks = NUM_BLOCKS;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr std::uint32_t icb0 = dfb::in0;
    constexpr std::uint32_t icb1 = dfb::in1;
    constexpr std::uint32_t ocb = dfb::out;

    compute_kernel_hw_startup(icb0, icb1, ocb);

#if PRIME_FULL_TILES
    {
        DataflowBuffer dfb2(dfb::in2);
        DataflowBuffer dfb3(dfb::in3);
        constexpr std::uint32_t icb2 = dfb::in2;
        constexpr std::uint32_t icb3 = dfb::in3;
        constexpr std::uint32_t prime_rt_dim = PRIME_RT_DIM;
        constexpr std::uint32_t prime_ct_dim = PRIME_CT_DIM;

        // The startup programmed the unpacker for the main operands' tile geometry; the priming
        // operands are full tiles, so swap the geometry in for the block and back out after it.
        reconfig_data_format(icb0, icb2, icb1, icb3);
        mul_bcast_cols_init_short_custom(icb2, icb3);
        dfb3.wait_front(prime_rt_dim);
        mul_bcast_col_block<false>(dfb2, dfb_out, icb2, icb3, ocb, prime_rt_dim, prime_ct_dim);
        dfb3.pop_front(prime_rt_dim);
        reconfig_data_format(icb2, icb0, icb3, icb1);
    }
#endif

    mul_bcast_cols_init_short_custom(icb0, icb1);

    // One bcast tile per row of a block, held for the whole run and never popped per block, so
    // tile_index_b indexes straight off the read pointer.
    dfb1.wait_front(rt_dim);

    for (std::uint32_t block = 0; block < num_blocks; block++) {
        mul_bcast_col_block<ACCUMULATE_ROWS != 0>(dfb0, dfb_out, icb0, icb1, ocb, rt_dim, ct_dim);
    }

    dfb1.pop_front(rt_dim);
}
