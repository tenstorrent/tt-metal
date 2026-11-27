// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// W8 unit 2 (GLM53_PF_MOE_XFFN_GANG, w8/combine_fusion_design.md) -- compute of the gang EP expert FFN: gate|up and the
// activation exactly as moe_xffn_compute.cpp on this core's W slice; the down phase over the 2 G N chunks in the ring
// order (moe_xffn_gang_reader.cpp), step q = hop h = q / 2 of the chunk's ring:
//   h = 0     : Y_s = A_s @ Wd_s                   -> CB_YRM as bf16 TILES (the writer sends them to the successor)
//   0 < h < G-1: Y = RING_IN (the predecessor's partial, reloaded into DST) + A_s @ Wd_s -> CB_YRM tiles (sent on)
//   h = G - 1 : the owner: RING_IN + A_s @ Wd_s = the full y block -> CB_YRM UNTILIZED rows (the writer writes y once)
// The reload is the bmm partials pattern (copy into DST, then matmul_block accumulates on top), fp32 DST.
// Compile-time: as moe_xffn_compute.cpp (W_T = the slice). Define XFFN_GANG = G.
#include <cstdint>

#include "api/compute/clamped_silu_glu.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "moe_xffn_common.hpp"

#ifndef XFFN_GANG
#error "the gang compute needs XFFN_GANG"
#endif

using namespace xffn;

// the CB holding K chunk kc's x rows: CB_XRM, or with XFFN_X_SPLIT CB_XRM (even chunks, reader) / CB_XRM1 (odd, writer)
FORCE_INLINE uint32_t xrm_cb(uint32_t kc) {
#ifdef XFFN_X_SPLIT
    return (kc & 1) ? CB_XRM1 : CB_XRM;
#else
    (void)kc;
    return CB_XRM;
#endif
}

void kernel_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(0);
    constexpr uint32_t W_T = get_compile_time_arg_val(1);
    constexpr uint32_t KC = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t MG = get_compile_time_arg_val(4);
    constexpr uint32_t SUBW = get_compile_time_arg_val(5);
    constexpr uint32_t SUBA = get_compile_time_arg_val(6);
    constexpr uint32_t GU_T = 2 * W_T;
    static_assert(GU_T % SUBA == 0 && NC % SUBW == 0 && SUBW <= 4 && SUBA <= 8, "fp32 DEST subblocks");

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_XT, CB_WGU, CB_GUP);
    while (true) {
        cb_wait_front(CB_CTL, 1);
        const uint32_t mt = read_tile_value(CB_CTL, 0, 0);
        cb_pop_front(CB_CTL, 1);
        if (mt == 0) {
            break;
        }

#ifdef XFFN_NO_MATH  // diagnostic: the data movement alone (every CB consumed / produced, no math)
        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
            const uint32_t cbx = xrm_cb(kc);
            cb_wait_front(cbx, MG * KC);
            cb_pop_front(cbx, MG * KC);
            cb_wait_front(CB_WGU, KC * GU_T);
            cb_pop_front(CB_WGU, KC * GU_T);
        }
#ifndef XFFN_NO_DOWN
        for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
            cb_wait_front(CB_WD, W_T * NC);
            for (uint32_t r = 0; r < mt; ++r) {
                cb_reserve_back(CB_YRM, NC);
                cb_push_back(CB_YRM, NC);
            }
            cb_pop_front(CB_WD, W_T * NC);
        }
#endif
        continue;
#endif
        // ---- A: gate|up, fp32 partials accumulated in L1 over the K chunks. Every CB below moves in whole MG-row
        // blocks (pushes must land exactly on the CB end); only the first mt tile rows carry data.
        cb_reserve_back(CB_GUP, MG * GU_T);
        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
            // one tile row = KC tile-sized pages of the chunk's x CB holding its 32 rows x KC tiles ROW_MAJOR
            const uint32_t cbx = xrm_cb(kc);
            cb_reserve_back(CB_XT, MG * KC);
            reconfig_data_format_srca(cbx);
            pack_reconfig_data_format(CB_XT);
            tilize_init(cbx, KC, CB_XT);
            for (uint32_t r = 0; r < mt; ++r) {
                cb_wait_front(cbx, KC);
                tilize_block(cbx, KC, CB_XT, 0, r * KC);
                cb_pop_front(cbx, KC);
            }
            tilize_uninit(cbx, CB_XT);
            if (mt < MG) {  // the producer's unused tile rows of this chunk
                cb_wait_front(cbx, (MG - mt) * KC);
                cb_pop_front(cbx, (MG - mt) * KC);
            }
            cb_push_back(CB_XT, MG * KC);

            cb_wait_front(CB_XT, MG * KC);
            cb_wait_front(CB_WGU, KC * GU_T);
            reconfig_data_format(CB_WGU, CB_XT);
            pack_reconfig_data_format(CB_GUP);
            matmul_block_init(CB_XT, CB_WGU, false, SUBA, 1, KC);
            PACK((llk_pack_reconfig_l1_acc(kc > 0 ? 1 : 0)));
            for (uint32_t r = 0; r < mt; ++r) {
                for (uint32_t nb = 0; nb < GU_T / SUBA; ++nb) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < KC; ++k) {
                        matmul_block(CB_XT, CB_WGU, r * KC + k, k * GU_T + nb * SUBA, 0, false, SUBA, 1, KC);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t s = 0; s < SUBA; ++s) {
                        pack_tile<true>(s, CB_GUP, r * GU_T + nb * SUBA + s);
                    }
                    tile_regs_release();
                }
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_pop_front(CB_WGU, KC * GU_T);
            cb_pop_front(CB_XT, MG * KC);
        }
        cb_push_back(CB_GUP, MG * GU_T);

        // ---- act: A = silu(min(G, 10)) * clamp(U, -10, 10), (gate_j, up_j) = partial tiles (2j, 2j + 1)
        cb_wait_front(CB_GUP, MG * GU_T);
        cb_reserve_back(CB_ACT, MG * W_T);
        reconfig_data_format_srca(CB_GUP);
        pack_reconfig_data_format(CB_ACT);
        copy_init(CB_GUP);
        clamped_silu_glu_tile_init();
        for (uint32_t r = 0; r < mt; ++r) {
            for (uint32_t j = 0; j < W_T; ++j) {
                tile_regs_acquire();
                copy_tile(CB_GUP, r * GU_T + 2 * j, 0);
                copy_tile(CB_GUP, r * GU_T + 2 * j + 1, 1);
                clamped_silu_glu_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, CB_ACT, r * W_T + j);
                tile_regs_release();
            }
        }
        cb_push_back(CB_ACT, MG * W_T);
        cb_pop_front(CB_GUP, MG * GU_T);

#ifdef XFFN_NO_DOWN  // diagnostic: gate|up + act only
        cb_wait_front(CB_ACT, MG * W_T);
        cb_pop_front(CB_ACT, MG * W_T);
        continue;
#endif
        // ---- B: down over the ring steps (see the header)
        constexpr uint32_t G = XFFN_GANG;
        static_assert(H_T / NC == 2 * G, "gang chunking: H / NC = 2 G N chunks");
        cb_wait_front(CB_ACT, MG * W_T);
        reconfig_data_format(CB_WD, CB_ACT);
        pack_reconfig_data_format(CB_YRM);
        matmul_block_init(CB_ACT, CB_WD, false, SUBW, 1, W_T);
        for (uint32_t q = 0; q < 2 * G; ++q) {
            const uint32_t h = q >> 1;
            const bool owner = h == G - 1;
            if (q == 2 * (G - 1)) {  // the owner's two chunks come last: untilized rows from here on
                pack_untilize_dest_init<SUBW, NC>(CB_YRM);
            }
            cb_wait_front(CB_WD, W_T * NC);
            for (uint32_t r = 0; r < mt; ++r) {
                if (h > 0) {
                    cb_wait_front(CB_RIN, NC);
                }
                cb_reserve_back(CB_YRM, NC);
                for (uint32_t sb = 0; sb < NC / SUBW; ++sb) {
                    tile_regs_acquire();
                    if (h > 0) {  // the travelling partial into DST; the matmul accumulates on top of it
                        reconfig_data_format_srca(CB_WD, CB_RIN);
                        copy_init(CB_RIN);
                        for (uint32_t t = 0; t < SUBW; ++t) {
                            copy_tile(CB_RIN, sb * SUBW + t, t);
                        }
                        reconfig_data_format_srca(CB_RIN, CB_WD);
                        matmul_block_init(CB_ACT, CB_WD, false, SUBW, 1, W_T);
                    }
                    for (uint32_t k = 0; k < W_T; ++k) {
                        matmul_block(CB_ACT, CB_WD, r * W_T + k, k * NC + sb * SUBW, 0, false, SUBW, 1, W_T);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    if (owner) {
                        pack_untilize_dest<SUBW, NC>(CB_YRM, 1, sb);
                    } else {
                        for (uint32_t t = 0; t < SUBW; ++t) {
                            pack_tile<true>(t, CB_YRM, sb * SUBW + t);
                        }
                    }
                    tile_regs_release();
                }
                if (h > 0) {
                    cb_pop_front(CB_RIN, NC);  // before the push: the writer credits the predecessor on YRM's front
                }
                cb_push_back(CB_YRM, NC);
            }
            cb_pop_front(CB_WD, W_T * NC);
        }
        pack_untilize_uninit(CB_YRM);
        cb_pop_front(CB_ACT, MG * W_T);
    }
}
