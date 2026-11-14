// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview shared-expert chain TRISC (8 DRAM-bank cores, core b). Copy of the DSV4 sk_sh_compute.cpp with the 9th tile pair removed:
// Hy4 has 256 intermediate columns per mesh column = exactly 8 tiles = one gate tile and one up tile per bank core.
//   phase A: x [K = KT tiles] x (gate column tile b, up column tile b) -> cb_gu = [g_b, u_b]
//            act_b = silu(min(g_b, L)) * clamp(u_b, -L, L)  -> cb_actl (broadcast by BRISC to slot b of every core's cb_act)
//   phase C: cb_act [8 tiles] x down columns [PNC b, PNC b + PNC) -> cb_o
// CT args (23 = EPI: 1 = swiglu on the accumulators for MT == 1, T <= 8): 0 cb_x, 1 cb_w, 2 cb_gu, 3 cb_gu16, 4 cb_actl, 5 cb_actl16, 6 cb_act, 7 cb_o, 8 cb_zero, 9 cb_zero16, 10 cb_tmp,
// 11 cb_tmp16, 12 KT, 13 fp32, 14 +L bits, 15 -L bits, 16 -BIG bits, 17 SBK, 18 NSUBK, 19 KA (8), 20 PNC, 21 SBWC, 22 MT

#include <cstdint>
#include "hsk_mm.hpp"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/debug/kernel_profiler.hpp"

template <uint32_t CB_IN16, uint32_t CB_ZERO16, uint32_t CB_TMP16>
FORCE_INLINE void sk_ew_group(uint32_t i0, uint32_t lo_bits, uint32_t hi_bits, bool do_silu, uint32_t slot) {
    tile_regs_acquire();
    add_init(CB_IN16, CB_ZERO16, false);
    add_tiles(CB_IN16, CB_ZERO16, i0, 0, 0);
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
    constexpr uint32_t cb_act = get_compile_time_arg_val(6);
    constexpr uint32_t cb_o = get_compile_time_arg_val(7);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(8);
    constexpr uint32_t cb_zero16 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(10);
    constexpr uint32_t cb_tmp16 = get_compile_time_arg_val(11);
    constexpr uint32_t KT = get_compile_time_arg_val(12);
    constexpr bool fp32 = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t lim_bits = get_compile_time_arg_val(14);
    constexpr uint32_t nlim_bits = get_compile_time_arg_val(15);
    constexpr uint32_t nbig_bits = get_compile_time_arg_val(16);
    constexpr uint32_t SBK = get_compile_time_arg_val(17);
    constexpr uint32_t NSUBK = get_compile_time_arg_val(18);
    constexpr uint32_t KA = get_compile_time_arg_val(19);
    constexpr uint32_t PNC = get_compile_time_arg_val(20);
    constexpr uint32_t SBWC = get_compile_time_arg_val(21);
    constexpr uint32_t MT = get_compile_time_arg_val(22);
    constexpr bool EPI = get_compile_time_arg_val(23) != 0 && MT == 1;

    deepseek_compute_kernel_init();

    cb_wait_front(cb_x, KT * MT);
    if constexpr (EPI) {
        DeviceZoneScopedN("T_MMA");
        sk_mm_epi<cb_x, cb_w, cb_actl, SBK, 2, 2, NSUBK, MT, fp32, 1>(0, KT, [&]() {
            clamp_tile_init();
            clamp_tile(0, nbig_bits, lim_bits);
            clamp_tile(1, nlim_bits, lim_bits);
            silu_tile_init();
            silu_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
        });
        cb_pop_front(cb_x, KT * MT);
    } else {
        {
            DeviceZoneScopedN("T_MMA");
            sk_mm<cb_x, cb_w, cb_gu, SBK, 2, 2, NSUBK, MT, fp32>(0, KT);
        }
        cb_pop_front(cb_x, KT * MT);
        {
            DeviceZoneScopedN("T_SWA");
            cb_wait_front(cb_gu, 2 * MT);
            cb_wait_front(cb_zero, 1);
            cb_reserve_back(cb_actl, MT);
            for (uint32_t m = 0; m < MT; ++m) {
                cb_reserve_back(cb_tmp, 2);
                reconfig_full_operand(cb_gu16, cb_zero16);
                pack_reconfig_data_format<true>(cb_tmp16);
                sk_ew_group<cb_gu16, cb_zero16, cb_tmp16>(2 * m, nbig_bits, lim_bits, true, 0);
                sk_ew_group<cb_gu16, cb_zero16, cb_tmp16>(2 * m + 1, nlim_bits, lim_bits, false, 1);
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
    }
    {
        DeviceZoneScopedN("T_MMC");
        cb_wait_front(cb_act, KA * MT);
        sk_mm<cb_act, cb_w, cb_o, KA, PNC, SBWC, 1, MT, fp32>(0, KA);
        cb_pop_front(cb_act, KA * MT);
    }
}
