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
    // Tiles broadcast per DEST section. Above 1, tile t is read at index t and lands at DEST slot 2 * t + 1, as in the
    // binary_ng SFPU broadcast kernels.
    constexpr uint32_t tiles_per_section = get_arg(args::tiles_per_section);
    static_assert(
        per_core_block_dim % tiles_per_section == 0, "per_core_block_dim must be a multiple of tiles_per_section");
    DataflowBuffer dfb_src(dfb::src);
    DataflowBuffer dfb_dst(dfb::dst);
    const uint32_t icb = dfb_src.get_id();
    const uint32_t ocb = dfb_dst.get_id();

    compute_kernel_hw_startup(icb, ocb);
    unary_bcast_init<BCAST_DIM>(icb);
    auto dst_slot = [](uint32_t tile_index, uint32_t t) { return tiles_per_section == 1 ? tile_index : 2 * t + 1; };

    // TODO (tt-metal #42792): revert to batched multi-tile broadcast once Quasar unpack<->pack semaphores land.
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; tile_index += tiles_per_section) {
            dfb_src.wait_front(tiles_per_section);
            dfb_dst.reserve_back(tiles_per_section);
            tile_regs_acquire();
            for (uint32_t t = 0; t < tiles_per_section; ++t) {
                unary_bcast<BCAST_DIM>(icb, t, dst_slot(tile_index, t));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < tiles_per_section; ++t) {
                pack_tile(dst_slot(tile_index, t), ocb);
            }
            tile_regs_release();
            dfb_src.pop_front(tiles_per_section);
            dfb_dst.push_back(tiles_per_section);
        }
    }
#else
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    unary_bcast_init<BCAST_DIM>(tt::CBIndex::c_0);

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
