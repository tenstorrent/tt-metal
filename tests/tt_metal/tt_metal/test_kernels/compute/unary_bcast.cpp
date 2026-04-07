// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// Host `defines` (e.g. BCAST_DIM) are emitted into defines_generated.h by the JIT; chlkc prolog includes it
// before this file on some paths. Pull it here when missing so template args see the macro (-I.. is the kernel out dir).
#ifndef BCAST_DIM
#include "defines_generated.h"
#endif
#include "api/compute/bcast.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

#ifdef ARCH_QUASAR
    constexpr uint32_t src_dfb_id = 0;
    constexpr uint32_t dst_dfb_id = 1;

    experimental::DataflowBuffer dfb_src(src_dfb_id);
    experimental::DataflowBuffer dfb_dst(dst_dfb_id);

    unary_bcast_init<BCAST_DIM>(src_dfb_id, dst_dfb_id);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            dfb_src.wait_front(1);
            dfb_dst.reserve_back(1);
            acquire_dst();
            // One tile visible per iteration: unpack always reads fifo front (index 0); dst slot is tile_index.
            unary_bcast<BCAST_DIM>(src_dfb_id, 0, tile_index);
            pack_tile(tile_index, dst_dfb_id);
            release_dst();
            dfb_src.pop_front(1);
            dfb_dst.push_back(1);
        }
    }
#else
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    unary_bcast_init<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb0.wait_front(1);
            cb16.reserve_back(1);
            acquire_dst();
            unary_bcast<BCAST_DIM>(tt::CBIndex::c_0, 0, tile_index);
            pack_tile(tile_index, tt::CBIndex::c_16);
            release_dst();
            cb0.pop_front(1);
            cb16.push_back(1);
        }
    }
#endif
}
