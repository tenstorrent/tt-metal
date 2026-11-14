// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// X40 TP32 routed experts, TRISC of bank core b.
//   meta: na (active experts), nj (experts this core reduces)
//   phase A: per active expert  x[K slice 24 tiles] @ block(96 tiles = 24 k x [g0 g1 u0 u1]) -> P (4 tiles, bf16)
//   (sk_mm, CT = 4) reduce : per reduced expert j, column c in {0,1}: g = sum over the 8 banks of slot c, u = sum of
//   slot 2 + c;
//            act_c = silu(min(g, L)) * clamp(u, -L, L) * score -> cb_actl (2 tiles per expert)
//   phase C: per sub-block s (6): 4 output tiles = sum over active experts of act_a @ block(8 tiles) (tp_down)
// CT: 0 cb_x, 1 cb_w, 2 cb_p, 3 cb_pr (tiny), 4 cb_pr16, 5 cb_sc (tiny), 6 cb_sc16, 7 cb_actl (tiny), 8 cb_actl16, 9
// cb_act, 10 cb_o, 11 cb_meta, 12 fp32, 13 +L bits, 14 -L bits, 15 -BIG bits
#include <cstdint>
#include "tp_mm.hpp"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_p = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pr = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pr16 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sc16 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_actl = get_compile_time_arg_val(7);
    constexpr uint32_t cb_actl16 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_act = get_compile_time_arg_val(9);
    constexpr uint32_t cb_o = get_compile_time_arg_val(10);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(11);
    constexpr bool fp32 = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t lim_bits = get_compile_time_arg_val(13);
    constexpr uint32_t nlim_bits = get_compile_time_arg_val(14);
    constexpr uint32_t nbig_bits = get_compile_time_arg_val(15);
    constexpr uint32_t DBG = get_compile_time_arg_val(16);

    deepseek_compute_kernel_init();

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t na = meta[0];
    const uint32_t nj = meta[1];

    // per group of 8 experts: phase A, then (if this core reduces expert j = g) the reduction + swiglu + score of that
    // expert
    cb_wait_front(cb_x, 24);
    if (nj > 0) {
        cb_wait_front(cb_sc, nj);
    }
    const uint32_t ng = (na + 7) / 8;
    for (uint32_t g = 0; g < ng; ++g) {
        const uint32_t n = (na - 8 * g < 8) ? (na - 8 * g) : 8;
        for (uint32_t a = 0; a < n; ++a) {
            sk_mm<cb_x, cb_w, cb_p, 24, 4, 4, 1, 1, fp32, 4>(0, 0);
        }
        if (DBG == 1) {
            continue;
        }
        if (g < nj) {
            const uint32_t j = g;
            cb_wait_front(cb_pr, 32 * (j + 1));
            reconfig_full_operand(cb_pr16, cb_pr16);
            pack_reconfig_data_format<true>(cb_actl16);
            cb_reserve_back(cb_actl, 2);
            for (uint32_t c = 0; c < 2; ++c) {
                tile_regs_acquire();
                const uint32_t pb = (j * 8) * 4;
                add_init(cb_pr16, cb_pr16, false);
                add_tiles(cb_pr16, cb_pr16, pb + c, pb + 4 + c, 0);
                add_tiles(cb_pr16, cb_pr16, pb + 2 + c, pb + 4 + 2 + c, 1);
                for (uint32_t q = 1; q < 4; ++q) {
                    add_tiles(cb_pr16, cb_pr16, pb + 8 * q + c, pb + 8 * q + 4 + c, 2);
                    add_tiles(cb_pr16, cb_pr16, pb + 8 * q + 2 + c, pb + 8 * q + 4 + 2 + c, 3);
                    add_binary_tile_init();
                    add_binary_tile(0, 2, 0);
                    add_binary_tile(1, 3, 1);
                    if (q < 3) {
                        add_init(cb_pr16, cb_pr16, false);
                    }
                }
                clamp_tile_init();
                clamp_tile(0, nbig_bits, lim_bits);
                clamp_tile(1, nlim_bits, lim_bits);
                silu_tile_init();
                silu_tile(0);
                copy_init<DST_ACCUM_MODE>(cb_sc16);
                copy_tile(cb_sc16, j, 2);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                mul_binary_tile(0, 2, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_actl16, (j & 1) * 2 + c);  // alias base is fixed, the 4-page ring holds 2 experts
                tile_regs_release();
            }
            cb_push_back(cb_actl, 2);
        }
    }
    cb_pop_front(cb_x, 24);
    if (DBG == 1) {
        return;
    }
    if (nj > 0) {
        cb_pop_front(cb_pr, nj * 32);
        cb_pop_front(cb_sc, nj);
    }
    if (DBG == 2) {
        return;
    }
    // phase C
    cb_wait_front(cb_act, 2 * na);
    tp_down<cb_act, cb_w, cb_o, 6, fp32>(na);
    cb_pop_front(cb_act, 2 * na);
    cb_pop_front(cb_meta, 1);
}
