// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// Host `defines` (e.g. BCAST_DIM) are emitted into defines_generated.h by the JIT; chlkc prolog includes it
// before this file on some paths. Pull it here when missing so template args see the macro (-I.. is the kernel out
// dir).
#ifndef BCAST_DIM
#include "defines_generated.h"
#endif
#include "api/compute/bcast.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);
    DataflowBuffer dfb_src(dfb::src);
    DataflowBuffer dfb_dst(dfb::dst);
    const uint32_t icb = dfb_src.get_id();
    const uint32_t ocb = dfb_dst.get_id();

    unary_bcast_init<BCAST_DIM>(icb, ocb);

    // TODO (tt-metal #42792): revert to batched multi-tile broadcast once Quasar unpack<->pack semaphores land.
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            dfb_src.wait_front(1);
            dfb_dst.reserve_back(1);
            tile_regs_acquire();
            // One tile visible per iteration: unpack always reads fifo front (index 0); dst slot is tile_index.
            unary_bcast<BCAST_DIM>(icb, 0, tile_index);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(tile_index, ocb);
            tile_regs_release();
            dfb_src.pop_front(1);
            dfb_dst.push_back(1);
        }
    }
#else
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    unary_bcast_init<BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // TODO (tt-metal #42792): revert to batched multi-tile broadcast once Quasar unpack<->pack semaphores land.
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb0.wait_front(1);
            cb16.reserve_back(1);
            tile_regs_acquire();
            unary_bcast<BCAST_DIM>(tt::CBIndex::c_0, 0, tile_index);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(tile_index, tt::CBIndex::c_16);
            tile_regs_release();
            cb0.pop_front(1);
            cb16.push_back(1);
        }
    }
#endif
}
