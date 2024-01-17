/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace ckernel {

ALWI void mul_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    mul_tiles_init();
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void copy_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    copy_tile_init();
    copy_tile(icb, itile, dst0);
    pack_tile(dst0, ocb);

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void add_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    add_tiles_init();
    add_tiles(icb0, icb1, itile0, itile1, dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    sub_tiles_init();
    sub_tiles(icb0, icb1, itile0, itile1, dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

}  // namespace ckernel
