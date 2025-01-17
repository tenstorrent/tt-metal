// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint.h"
#include "debug/ring_buffer.h"

// PACK trisc version of cb_wait_front just for this test
#if defined(UCK_CHLKC_PACK)
#include "compute_kernel_api/cb_api.h"
inline void cb_wait_front_pack(int operand, std::int32_t num_tiles) {
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    do {
        tiles_received = (std::uint16_t)reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - get_local_cb_interface(input).tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
}
#endif

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/untilize.h"
ALWI void UNTILIZE_TILES(uint32_t in0_cb, uint32_t out_cb, uint32_t num_tiles) {
    untilize_init(in0_cb, out_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    untilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    untilize_uninit(in0_cb);
}
namespace NAMESPACE {
void MAIN {
    // Read out the tile we want to print using BRISC, put it in c_in0
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_1;
    uint32_t is_tilized = get_arg_val<uint32_t>(0);

    // PACK trisc doesn't have cb_wait_front implemented, have our own version just for this test.
#if defined(UCK_CHLKC_PACK)
    cb_wait_front_pack(cb_id, 1);
#else
    cb_wait_front(cb_id, 1);
#endif

    // For tilized formats, also test untilizing them on device and make sure we can print.
    if (is_tilized) {
        UNTILIZE_TILES(cb_id, cb_intermed, 1);
    }
    // Print the tile from each RISC, one after another
    DPRINT_UNPACK(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{1}; DPRINT << "Print tile from Unpack:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, is_tilized) << ENDL();
        DPRINT << RAISE{2};);
    DPRINT_MATH(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{2}; DPRINT << "Print tile from Math:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, is_tilized) << ENDL();
        DPRINT << RAISE{3};);
    DPRINT_PACK(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{3}; DPRINT << "Print tile from Pack:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, is_tilized) << ENDL();
        DPRINT << RAISE{4};);
}
}  // namespace NAMESPACE
