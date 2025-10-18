// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/fused_eltwise_binary_reduce_multiple_tiles.h"

#include "tt_metal/hw/inc/debug/dprint_tensix.h"
#include "tt_metal/hw/inc/debug/dprint_pages.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    uint32_t tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_inp0 = tt::CBIndex::c_0;
    constexpr auto cb_inp1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    cb_wait_front(cb_inp0, tile_cnt);
    cb_wait_front(cb_inp1, tile_cnt);

    // UNPACK(tt::compute::common::print_full_tile(cb_inp1, 0, true));

    cb_reserve_back(cb_out0, 1);

    fused_eltwise_binary_reduce_init<ELTWISE_OP_TYPE, REDUCE_OP, REDUCE_DIM>(cb_inp0, cb_inp1);

    tile_regs_acquire();

    fused_eltwise_binary_reduce<ELTWISE_OP_TYPE, REDUCE_OP, REDUCE_DIM>(cb_inp0, cb_inp1, 0, 0, tile_cnt);

    tile_regs_commit();
    tile_regs_wait();

    cb_pop_front(cb_inp0, tile_cnt);
    cb_pop_front(cb_inp1, tile_cnt);

    pack_tile(0, cb_out0);  // Result is always in tile 0 after reduce operation

    // Add some NOPs to ensure pack is done before we push/print the tile
    for (uint32_t i = 0; i < 32; ++i) {
        PACK(TTI_NOP);
    }

    cb_push_back(cb_out0, 1);

    // Print the output tile for debugging
    // tt::compute::common::print_full_tile(cb_out0, 0, true);

    tile_regs_release();

    fused_eltwise_binary_reduce_uninit();
}
}  // namespace NAMESPACE
