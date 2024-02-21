/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace ckernel {

// TODO(seunghwan100): If p is 2 and decimal is 0, we can use sqrt_tile.
ALWI void power_tile_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ACQ();
    cb_wait_front(cb_x, onetile);
    cb_reserve_back(cb_xpow, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }

    pack_tile(dst0, cb_xpow);

    cb_push_back(cb_xpow, onetile);
    REL();
    // We don't pop cb_x here.

    // log(x)
    ACQ();
    cb_reserve_back(cb_logx, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);

    pack_tile(dst0, cb_logx);

    cb_pop_front(cb_x, onetile);
    cb_push_back(cb_logx, onetile);
    REL();

    // exp(log(x) * decimal)
    ACQ();
    cb_wait_front(cb_logx, onetile);
    cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init();
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);

    pack_tile(dst0, cb_exp_lxmd);

    cb_pop_front(cb_logx, onetile);
    cb_push_back(cb_exp_lxmd, onetile);
    REL();

    // x^p * exp(log(x) * decimal)(==(x + decimal)^p)
    ACQ();
    cb_wait_front(cb_xpow, onetile);
    cb_wait_front(cb_exp_lxmd, onetile);
    cb_reserve_back(cb_correct_xpow, onetile);

    mul_tiles_init();
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

    pack_tile(dst0, cb_correct_xpow);

    cb_pop_front(cb_xpow, onetile);
    cb_pop_front(cb_exp_lxmd, onetile);
    cb_push_back(cb_correct_xpow, onetile);
    REL();
}

ALWI void power_and_recip_tile_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_recip_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ACQ();
    cb_wait_front(cb_x, onetile);
    cb_reserve_back(cb_xpow, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }

    pack_tile(dst0, cb_xpow);

    cb_push_back(cb_xpow, onetile);
    REL();
    // We don't pop cb_x here.

    // log(x)
    ACQ();
    cb_reserve_back(cb_logx, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);

    pack_tile(dst0, cb_logx);

    cb_pop_front(cb_x, onetile);
    cb_push_back(cb_logx, onetile);
    REL();

    // exp(log(x) * decimal)
    ACQ();
    cb_wait_front(cb_logx, onetile);
    cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init();
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);

    pack_tile(dst0, cb_exp_lxmd);

    cb_pop_front(cb_logx, onetile);
    cb_push_back(cb_exp_lxmd, onetile);
    REL();

    // 1 / (x^p * exp(log(x) * decimal))(==1 / (x + decimal)^p)
    ACQ();
    cb_wait_front(cb_xpow, onetile);
    cb_wait_front(cb_exp_lxmd, onetile);
    cb_reserve_back(cb_recip_xpow, onetile);

    mul_tiles_init();
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

    recip_tile_init();
    recip_tile(dst0);

    pack_tile(dst0, cb_recip_xpow);

    cb_pop_front(cb_xpow, onetile);
    cb_pop_front(cb_exp_lxmd, onetile);
    cb_push_back(cb_recip_xpow, onetile);
    REL();
}

}  // namespace ckernel
