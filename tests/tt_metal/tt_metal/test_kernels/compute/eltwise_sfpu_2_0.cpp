// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// DEST placement knobs of the unary path (QuasarUnpackToDestDestIndex). DST_TILE_START: DEST index the tile lands
// at and is packed back from. SECTION_NTILES: one DEST section of that many tiles per tile_regs_acquire instead of
// one tile, placed with one copy_block or, with SECTION_USE_COPY_TILE, one copy_tile per tile, and packed back from
// the same indices. Section mode runs no SFPU chain: every chain in this file targets DEST tile 0.
#ifndef DST_TILE_START
#define DST_TILE_START 0
#endif
#if defined(SECTION_NTILES) && (defined(SFPU_OP_CHAIN_0) || defined(SFPU_BINARY_OP) || defined(SFPU_TERNARY_OP))
#error "SECTION_NTILES is a unary datacopy mode: the SFPU chains target DEST tile 0"
#endif

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    [[maybe_unused]] const uint32_t per_core_block_size = get_arg(args::per_core_block_size);  // unused in section mode

    // DFBs — arity selected at compile time.
#if defined(SFPU_TERNARY_OP)
    DataflowBuffer dfb_in0(dfb::in0), dfb_in1(dfb::in1), dfb_in2(dfb::in2);
#elif defined(SFPU_BINARY_OP)
    DataflowBuffer dfb_in0(dfb::in0), dfb_in1(dfb::in1);
#else  // SFPU_UNARY_OP (default)
    DataflowBuffer dfb_in0(dfb::in);
#endif
    DataflowBuffer dfb_out(dfb::out);

    const uint32_t in0_id = dfb_in0.get_id();
    const uint32_t out_id = dfb_out.get_id();
#if defined(SFPU_BINARY_OP) || defined(SFPU_TERNARY_OP)
    const uint32_t in1_id = dfb_in1.get_id();
#endif
#if defined(SFPU_TERNARY_OP)
    const uint32_t in2_id = dfb_in2.get_id();
#endif

    compute_kernel_hw_startup(in0_id, out_id);
    copy_init(in0_id);
#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
#if defined(SECTION_NTILES)
        // One DEST section per acquire: the tiles land at [DST_TILE_START, DST_TILE_START + SECTION_NTILES).
        dfb_in0.wait_front(SECTION_NTILES);
        dfb_out.reserve_back(SECTION_NTILES);

        tile_regs_acquire();
        copy_init(in0_id);
#if defined(SECTION_USE_COPY_TILE)
        for (uint32_t i = 0; i < SECTION_NTILES; ++i) {
            copy_tile(in0_id, /*tile_index=*/i, /*dst_index=*/DST_TILE_START + i);
        }
#else
        copy_block(in0_id, /*start_in_tile_index=*/0, /*start_dst_tile_index=*/DST_TILE_START, SECTION_NTILES);
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < SECTION_NTILES; ++i) {
            pack_tile(/*dst_index=*/DST_TILE_START + i, out_id);
        }
        tile_regs_release();

        dfb_out.push_back(SECTION_NTILES);
        dfb_in0.pop_front(SECTION_NTILES);
#else
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            dfb_in0.wait_front(1);
#if defined(SFPU_BINARY_OP) || defined(SFPU_TERNARY_OP)
            dfb_in1.wait_front(1);
#endif
#if defined(SFPU_TERNARY_OP)
            dfb_in2.wait_front(1);
#endif
            dfb_out.reserve_back(1);

            tile_regs_acquire();
            copy_init(in0_id);
            copy_tile(in0_id, /*tile_index=*/0, /*dst_index=*/DST_TILE_START);
#if defined(SFPU_BINARY_OP) || defined(SFPU_TERNARY_OP)
            copy_init(in1_id);
            copy_tile(in1_id, /*tile_index=*/0, /*dst_index=*/1);
#endif
#if defined(SFPU_TERNARY_OP)
            copy_init(in2_id);
            copy_tile(in2_id, /*tile_index=*/0, /*dst_index=*/2);
#endif
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();
            tile_regs_wait();
#if defined(SFPU_TERNARY_OP)
            pack_tile(/*dst_index=*/3, out_id);
#else
            pack_tile(/*dst_index=*/DST_TILE_START, out_id);
#endif
            tile_regs_release();

            dfb_out.push_back(1);
            dfb_in0.pop_front(1);
#if defined(SFPU_BINARY_OP) || defined(SFPU_TERNARY_OP)
            dfb_in1.pop_front(1);
#endif
#if defined(SFPU_TERNARY_OP)
            dfb_in2.pop_front(1);
#endif
        }
#endif  // SECTION_NTILES
    }
}
