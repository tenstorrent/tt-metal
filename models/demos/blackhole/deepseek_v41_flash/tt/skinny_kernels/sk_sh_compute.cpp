// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// shared-expert chain TRISC (8 DRAM-bank cores, core b):
//   phase A: x [K = 160 tiles] x (gate column tile b, up column tile b) -> cb_gu = [g_b, u_b]
//            act_b = silu(min(g_b, L)) * clamp(u_b, -L, L)                    -> cb_actl (broadcast to every core's
//            cb_act slot b by BRISC)
//   phase B: x K-slice [20 b, 20 b + 20) x (gate_8, up_8 K slices of the 9th tile pair) -> cb_p = [gp_b, up_b] (partial
//   sums, all-gathered into cb_prg / cb_pru slot b)
//            act_8 = silu(min(sum_c gp_c, L)) * clamp(sum_c up_c, -L, L)       -> cb_act8 (BRISC places it in cb_act
//            slot 8)
//   phase C: cb_act [9 tiles] x down columns [20 b, 20 b + 20) -> cb_o
// The element-wise parts run on 16x16 (one face) ALIAS views of the tiny 8x32 tile buffers (256 elements each, same
// memory): cb_*16. CT args: 0 cb_x, 1 cb_w, 2 cb_gu, 3 cb_gu16, 4 cb_actl, 5 cb_actl16, 6 cb_p, 7 cb_prg, 8 cb_prg16, 9
// cb_pru, 10 cb_pru16, 11 cb_act, 12 cb_act8, 13 cb_act8_16, 14 cb_o, 15 cb_zero, 16 cb_zero16, 17 cb_tmp, 18 cb_tmp16,
// 19 KT (x tiles), 20 fp32, 21 +L bits, 22 -L bits, 23 -BIG bits, 24 NC, 25 KSL (K slice tiles of the pair 8), 26 KA
// (act tiles = 9), 27 PNC (down columns per core), 28 SBWC 29 MT (row tiles: 1 for T <= 8, 2 for T = 9..16; buffers
// hold MT groups: x [m][KT], gu / p [m][2], prg / pru [m][NC], act [m][9]) RT args: 0 core index b

#include <cstdint>
#include "sk_mm.hpp"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/compute_kernel_api.h"
#include "api/debug/kernel_profiler.hpp"

// dest[0] = clamp(sum of the tiles [i0, i0 + 2 n) of cb (pairwise add_tiles into dest) or, n = 0, of the single tile
// i0), optional min / silu
template <uint32_t CB_IN16, uint32_t CB_ZERO16, uint32_t CB_TMP16>
FORCE_INLINE void sk_ew_group(
    uint32_t i0, uint32_t npairs, uint32_t lo_bits, uint32_t hi_bits, bool do_silu, uint32_t slot) {
    tile_regs_acquire();
    if (npairs == 0) {
        add_init(CB_IN16, CB_ZERO16, false);
        add_tiles(CB_IN16, CB_ZERO16, i0, 0, 0);
    } else {
        add_init(CB_IN16, CB_IN16, true);
        for (uint32_t i = 0; i < npairs; ++i) {
            add_tiles(CB_IN16, CB_IN16, i0 + 2 * i, i0 + 2 * i + 1, 0);
        }
    }
    clamp_tile_init();
    clamp_tile(0, lo_bits, hi_bits);
    if (do_silu) {
        silu_tile_init();
        silu_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, CB_TMP16, slot);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gu = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gu16 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_actl = get_compile_time_arg_val(4);
    constexpr uint32_t cb_actl16 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_p = get_compile_time_arg_val(6);
    constexpr uint32_t cb_prg = get_compile_time_arg_val(7);
    constexpr uint32_t cb_prg16 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_pru = get_compile_time_arg_val(9);
    constexpr uint32_t cb_pru16 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_act = get_compile_time_arg_val(11);
    constexpr uint32_t cb_act8 = get_compile_time_arg_val(12);
    constexpr uint32_t cb_act8_16 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_o = get_compile_time_arg_val(14);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(15);
    constexpr uint32_t cb_zero16 = get_compile_time_arg_val(16);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(17);
    constexpr uint32_t cb_tmp16 = get_compile_time_arg_val(18);
    constexpr uint32_t KT = get_compile_time_arg_val(19);
    constexpr bool fp32 = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t lim_bits = get_compile_time_arg_val(21);
    constexpr uint32_t nlim_bits = get_compile_time_arg_val(22);
    constexpr uint32_t nbig_bits = get_compile_time_arg_val(23);
    constexpr uint32_t NC = get_compile_time_arg_val(24);
    constexpr uint32_t KSL = get_compile_time_arg_val(25);
    constexpr uint32_t KA = get_compile_time_arg_val(26);
    constexpr uint32_t PNC = get_compile_time_arg_val(27);
    constexpr uint32_t SBWC = get_compile_time_arg_val(28);
    constexpr uint32_t MT = get_compile_time_arg_val(29);
    const uint32_t b = get_arg_val<uint32_t>(0);

    deepseek_compute_kernel_init();

    cb_wait_front(cb_x, KT * MT);
    {
        DeviceZoneScopedN("T_MMA");
        sk_mm<cb_x, cb_w, cb_gu, 40, 2, 2, KT / 40, MT, fp32>(0, KT);
    }
    {
        DeviceZoneScopedN("T_MMB");
        sk_mm<cb_x, cb_w, cb_p, KSL, 2, 2, 1, MT, fp32>(b * KSL, KT);
    }
    cb_pop_front(cb_x, KT * MT);
    // ---- act_b = silu(min(g, L)) * clamp(u, -L, L), per row tile ----
    {
        DeviceZoneScopedN("T_SWA");
        cb_wait_front(cb_gu, 2 * MT);
        cb_wait_front(cb_zero, 1);
        cb_reserve_back(cb_actl, MT);
        for (uint32_t m = 0; m < MT; ++m) {
            cb_reserve_back(cb_tmp, 2);
            reconfig_full_operand(cb_gu16, cb_zero16);
            pack_reconfig_data_format<true>(cb_tmp16);
            sk_ew_group<cb_gu16, cb_zero16, cb_tmp16>(2 * m, 0, nbig_bits, lim_bits, true, 0);
            sk_ew_group<cb_gu16, cb_zero16, cb_tmp16>(2 * m + 1, 0, nlim_bits, lim_bits, false, 1);
            cb_push_back(cb_tmp, 2);
            cb_wait_front(cb_tmp, 2);
            mul_init(cb_tmp16, cb_tmp16);
            tile_regs_acquire();
            mul_tiles(cb_tmp16, cb_tmp16, 0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format<true>(cb_actl16);
            pack_tile<true>(0, cb_actl16, m);
            tile_regs_release();
            cb_pop_front(cb_tmp, 2);
        }
        cb_push_back(cb_actl, MT);
        cb_pop_front(cb_gu, 2 * MT);
    }
    // ---- pair 8: reduce the NC partials, act_8, per row tile ----
    {
        DeviceZoneScopedN("T_SW8");
        cb_wait_front(cb_prg, NC * MT);
        cb_wait_front(cb_pru, NC * MT);
        cb_reserve_back(cb_act8, MT);
        for (uint32_t m = 0; m < MT; ++m) {
            cb_reserve_back(cb_tmp, 2);
            reconfig_full_operand(cb_prg16, cb_prg16);
            pack_reconfig_data_format<true>(cb_tmp16);
            sk_ew_group<cb_prg16, cb_zero16, cb_tmp16>(m * NC, NC / 2, nbig_bits, lim_bits, true, 0);
            sk_ew_group<cb_pru16, cb_zero16, cb_tmp16>(m * NC, NC / 2, nlim_bits, lim_bits, false, 1);
            cb_push_back(cb_tmp, 2);
            cb_wait_front(cb_tmp, 2);
            mul_init(cb_tmp16, cb_tmp16);
            tile_regs_acquire();
            mul_tiles(cb_tmp16, cb_tmp16, 0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format<true>(cb_act8_16);
            pack_tile<true>(0, cb_act8_16, m);
            tile_regs_release();
            cb_pop_front(cb_tmp, 2);
        }
        cb_push_back(cb_act8, MT);
        cb_pop_front(cb_prg, NC * MT);
        cb_pop_front(cb_pru, NC * MT);
    }
    // ---- phase C: down projection ----
    {
        DeviceZoneScopedN("T_MMC");
        cb_wait_front(cb_act, KA * MT);
        sk_mm<cb_act, cb_w, cb_o, KA, PNC, SBWC, 1, MT, fp32>(0, KA);
        cb_pop_front(cb_act, KA * MT);
    }
}
