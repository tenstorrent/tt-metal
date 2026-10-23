// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"
#include "api/debug/ring_buffer.h"

// PACK trisc version of cb_wait_front just for this test
#if defined(UCK_CHLKC_PACK)
#include "api/compute/cb_api.h"
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

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"

// Largest pack_untilize block width (<= DEST tile capacity) dividing full_ct_dim.
constexpr uint32_t untilize_pack_block_ct(uint32_t full_ct_dim) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (full_ct_dim % bct == 0) {
            return bct;
        }
    }
    return 1;
}

template <uint32_t num_tiles>
ALWI void UNTILIZE_TILES(uint32_t in0_cb, uint32_t out_cb) {
    constexpr uint32_t block_ct = untilize_pack_block_ct(num_tiles);
    constexpr uint32_t num_blocks = num_tiles / block_ct;
    compute_kernel_hw_startup(in0_cb, out_cb);
    pack_untilize_init<block_ct, num_tiles>(in0_cb, out_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        pack_untilize_block<block_ct, num_tiles>(in0_cb, 1, out_cb, b);
        cb_pop_front(in0_cb, block_ct);
    }
    cb_push_back(out_cb, num_tiles);
    pack_untilize_uninit(out_cb);
}
void kernel_main() {
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
        UNTILIZE_TILES<1>(cb_id, cb_intermed);
    }
    // Print the tile from each RISC, one after another
    DEVICE_PRINT_UNPACK("Print tile from Unpack:\n{}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, is_tilized));
    DEVICE_PRINT_MATH("Print tile from Math:\n{}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, is_tilized));
    DEVICE_PRINT_PACK("Print tile from Pack:\n{}\n", TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, is_tilized));
}
