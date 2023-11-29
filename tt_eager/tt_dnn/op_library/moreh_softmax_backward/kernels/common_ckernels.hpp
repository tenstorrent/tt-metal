/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/eltwise_unary/negative.h"

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

ALWI void mul_tiles_and_negative_to_cb(
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

    negative_tile_init();
    negative_tile(dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}



ALWI void mul_tiles_and_mask_tile_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t mtile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1,
    uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);
    cb_wait_front(maskcb, mtile + 1);

    mul_tiles_init();
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    constexpr int dst_mask = 1;
    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);
    if (popm)
        cb_pop_front(maskcb, popm);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_rows_to_cb(
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

    mul_bcast_rows_init_short();
    mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_cols_to_cb(
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

    mul_bcast_cols_init_short();
    mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, dst0);

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

ALWI void exp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    copy_tile_init();
    copy_tile(icb, itile, dst);

    exp_tile_init();
    exp_tile(dst);

    pack_tile(dst, ocb);

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void rexp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    copy_tile_init();
    copy_tile(icb, itile, dst);

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);

    pack_tile(dst, ocb);

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void exp_tile_and_mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);
    cb_wait_front(maskcb, mtile + 1);

    copy_tile_init();
    copy_tile(icb, itile, dst);

    if (pop)
        cb_pop_front(icb, pop);

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);

    if (popm)
        cb_pop_front(maskcb, popm);

    pack_tile(dst, ocb);

    cb_push_back(ocb, onetile);
}

ALWI void rexp_tile_and_mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);
    cb_wait_front(maskcb, mtile + 1);

    copy_tile_init();
    copy_tile(icb, itile, dst);

    if (pop)
        cb_pop_front(icb, pop);

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);

    if (popm)
        cb_pop_front(maskcb, popm);

    pack_tile(dst, ocb);

    cb_push_back(ocb, onetile);
}

ALWI void recip_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    copy_tile_init();
    copy_tile(icb, itile, dst0);

    recip_tile_init();
    recip_tile(dst0);

    pack_tile(dst0, ocb);

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void add_tiles_to_cb(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, onetile);
    cb_wait_front(icb1, onetile);

    add_tiles_init();
    add_tiles(icb0, icb1, 0, 0, dst0);

    pack_tile(dst0, ocb);

    cb_pop_front(icb0, onetile);
    cb_pop_front(icb1, onetile);

    cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 1,
    uint32_t itile1 = 1,
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

ALWI void sub_tiles_bcast_rows_to_cb(
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

    // sub_bcast_rows_init_short();
    {
        MATH((llk_math_eltwise_binary_init<ELWSUB, BroadcastType::ROW, MATH_FIDELITY>()));
#ifdef ARCH_GRAYSKULL
        UNPACK((llk_unpack_AB_init<BroadcastType::ROW>()));
#else
        UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(0, 1)));
#endif
    }
    sub_tiles_bcast<BroadcastType::ROW>(icb0, icb1, itile0, itile1, dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_bcast_cols_to_cb(
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

    sub_bcast_cols_init_short();
    sub_tiles_bcast<BroadcastType::COL>(icb0, icb1, itile0, itile1, dst0);

    pack_tile(dst0, ocb);

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void reduce_tile_to_cb(
    PoolType reduce_op,
    ReduceDim dim,
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t size,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb1, onetile);

    reduce_init_delta<false>(reduce_op, dim);
    for (uint32_t x = 0; x < size; ++x) {
        cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        reduce_tile(reduce_op, dim, icb0, icb1, x, bcast_scaler0, dst0);
    }
    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    reduce_revert_delta();
    pack_tile(dst0, ocb);
    cb_push_back(ocb, onetile);
}

ALWI void reduce_tile_and_recip_tile_to_cb(
    PoolType reduce_op,
    ReduceDim dim,
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t size,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb1, onetile);

    reduce_init_delta<false>(reduce_op, dim);
    for (uint32_t x = 0; x < size; ++x) {
        cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        reduce_tile(reduce_op, dim, icb0, icb1, x, bcast_scaler0, dst0);
    }
    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    reduce_revert_delta();

    recip_tile_init();
    recip_tile(dst0);

    pack_tile(dst0, ocb);
    cb_push_back(ocb, onetile);
}

}  // namespace ckernel
