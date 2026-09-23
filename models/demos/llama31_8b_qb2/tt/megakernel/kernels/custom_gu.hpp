// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/experimental/custom_mm.h"

#ifdef TRISC_UNPACK
// Use the pinned source revision's custom MOP with an explicit weight row
// stride. Native [K, Width] bank rows supply each output subblock without
// a persistent weight permutation or extra weight payload.
template <uint32_t A, uint32_t B, uint32_t Width, uint32_t Subblock, uint32_t KBlock>
void custom_projection_unpack(uint32_t column) {
    auto& weights = get_local_cb_interface(B);
    auto& input = get_local_cb_interface(A);
    volatile uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(1);
    reset_config_context();
    TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    _llk_unpack_AB_custom_mm_run_(cfg,
        weights.fifo_rd_ptr - 1 + column * weights.fifo_page_size,
        input.fifo_rd_ptr - 1, weights.fifo_page_size, (Width - Subblock + 1) * weights.fifo_page_size, KBlock);
}
#endif

template <uint32_t A, uint32_t B, uint32_t Out, uint32_t Partial, uint32_t KBlock, uint32_t Width, uint32_t K, uint32_t Subblock>
void custom_projection() {
    // A is 8 rows; partial/output stay 16 rows to retain face1 at DST row16.
    // Preserve original K blocks, BF16 L1 partials, LoFi, no split accumulation.
    constexpr uint32_t blocks = K / KBlock;
    static_assert(KBlock % 2 == 0);
    custom_mm_block_init_short<false, false, false>(A, B, Partial, Subblock);
    pack_reconfig_data_format(Partial);
    pack_reconfig_l1_acc(0);
    for (uint32_t block = 0; block < blocks; ++block) {
        cb_wait_front(A, KBlock);
        cb_wait_front(B, KBlock * Width);
        const bool last = block == blocks - 1;
#if PROJECTION_HOIST_PACK
        if (block == 1 || last) { pack_reconfig_l1_acc(!last); }
        if (last) { pack_reconfig_data_format(Out); }
#endif
        for (uint32_t column = 0; column < Width; column += Subblock) {
            tile_regs_acquire();
            if (last) {
                reconfig_data_format_srca(B, Partial);
                copy_init(Partial);
                cb_wait_front(Partial, Subblock);
                copy_block(Partial, 0, 0, Subblock);
                cb_pop_front(Partial, Subblock);
                reconfig_data_format_srca(Partial, B);
                custom_mm_block_init_short<false, false, false>(A, B, Partial, Subblock);
            }
            UNPACK((tensix_sync()));
            MATH((tensix_sync()));
            UNPACK((custom_projection_unpack<A, B, Width, Subblock, KBlock>(column)));
            MATH((llk_math_custom_mm<false>(A, B, 0, KBlock, Subblock)));
            tile_regs_commit();
            const uint32_t destination = last ? Out : Partial;
            cb_reserve_back(destination, Subblock);
            tile_regs_wait();
#if !PROJECTION_HOIST_PACK
            pack_reconfig_data_format(destination);
            pack_reconfig_l1_acc(!last && block > 0);
#endif
            pack_block(0, destination, Subblock);
            tile_regs_release();
            cb_push_back(destination, Subblock);
        }
        if (block < blocks - 2) {
            for (uint32_t column = 0; column < Width; column += Subblock) {
                cb_wait_front(Partial, Subblock);
                cb_pop_front(Partial, Subblock);
            }
        }
        cb_pop_front(A, KBlock);
        cb_pop_front(B, KBlock * Width);
    }
    pack_reconfig_l1_acc(0);
    custom_mm_block_uninit<false>();
}
