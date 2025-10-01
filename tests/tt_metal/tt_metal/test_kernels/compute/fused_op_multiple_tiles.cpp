// if you can't find it: reduce_h4.cpp - DO NOT REMOVE THIS COMMENT

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/fused_eltwise_binary_reduce_multiple_tiles.h"

#include "/localdev/vbabic/tt-metal/tt_metal/hw/inc/debug/dprint_tensix.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    uint32_t tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_inp0 = cb_in0;
    constexpr auto cb_inp1 = cb_in1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    cb_wait_front(cb_inp0, tile_cnt);
    cb_wait_front(cb_inp1, tile_cnt);
    cb_reserve_back(cb_out0, 1);

    fused_eltwise_binary_reduce_init<ELTWISE_OP_TYPE, REDUCE_OP, REDUCE_DIM>(cb_inp0, cb_inp1);

    tile_regs_acquire();

    fused_eltwise_binary_reduce<ELTWISE_OP_TYPE, REDUCE_OP, REDUCE_DIM>(cb_inp0, cb_inp1, 0, 0, tile_cnt);

    // dprint_tensix_dest_reg(0);

    tile_regs_commit();
    tile_regs_wait();

    cb_pop_front(cb_inp0, tile_cnt);
    cb_pop_front(cb_inp1, tile_cnt);

    pack_tile(0, cb_out0);  // Result is always in tile 0 after reduce operation

    PACK(for (uint32_t i = 0; i < 32; ++i) { TTI_NOP; });

    // DPRINT_PACK({
    //     DPRINT << "Output tile in cb_out0:" << ENDL();
    //     for (uint16_t r = 0; r < 32; ++r) {
    //         DPRINT << (uint)r << " : "
    //                << TileSlice(
    //                       cb_out0,
    //                       0,
    //                       SliceRange{
    //                           .h0 = (uint8_t)r,
    //                           .h1 = (uint8_t)(r + 1),
    //                           .hs = (uint8_t)1,
    //                           .w0 = (uint8_t)0,
    //                           .w1 = (uint8_t)32,
    //                           .ws = (uint8_t)1},
    //                       true,
    //                       false)
    //                << ENDL();
    //     }
    // });

    cb_push_back(cb_out0, 1);

    tile_regs_release();

    fused_eltwise_binary_reduce_uninit();
}
}  // namespace NAMESPACE
