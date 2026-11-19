// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 XSL expert FFN (moe_xffn_common.hpp) -- compute. Per M group (control page: mt tile rows; 0 = end):
//   A  per K chunk: tilize the mt x-row blocks, then for every tile row and gate|up subblock of SUBW
//      tiles matmul_block over the chunk's KC tiles, packed onto the group's fp32 partials (packer L1 accumulation
//      from the second chunk on; the partial tiles stay reserved -- never pushed -- until the last chunk);
//   act the partials unpacked exactly (fp32 unpack-to-dest) pairwise (gate_j, up_j) -> clamped_silu_glu -> bf16 A;
//   B  per N chunk of NC tiles: per tile row, SUBW-wide subblocks of matmul_block over the W tiles of A, each packed
//      UNTILIZED into the tile row's 32 x NC ROW_MAJOR block (pack_untilize_dest, block column = subblock).
// Compile-time: H_T, W_T, KC, NC, MG, SUBW (down / untilize subblock, <= 4), SUBA (gate|up subblock: 4, or 8 with
// full-sync DEST -- dst_full_sync_en on the host -- halving the matmul_block calls). fp32 DEST.
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
        // ---- B: down, untilized straight into the tile row's ROW_MAJOR block
        cb_wait_front(CB_ACT, MG * W_T);
        reconfig_data_format(CB_WD, CB_ACT);
        pack_reconfig_data_format(CB_YRM);
        pack_untilize_dest_init<SUBW, NC>(CB_YRM);
        matmul_block_init(CB_ACT, CB_WD, false, SUBW, 1, W_T);
        for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
            cb_wait_front(CB_WD, W_T * NC);
            for (uint32_t r = 0; r < mt; ++r) {
                cb_reserve_back(CB_YRM, NC);  // NC tile-sized pages = the tile row's 32 rows x NC tiles ROW_MAJOR
                for (uint32_t sb = 0; sb < NC / SUBW; ++sb) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < W_T; ++k) {
                        matmul_block(CB_ACT, CB_WD, r * W_T + k, k * NC + sb * SUBW, 0, false, SUBW, 1, W_T);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_untilize_dest<SUBW, NC>(CB_YRM, 1, sb);
                    tile_regs_release();
                }
                cb_push_back(CB_YRM, NC);
            }
            cb_pop_front(CB_WD, W_T * NC);
        }
        pack_untilize_uninit(CB_YRM);
        cb_pop_front(CB_ACT, MG * W_T);
    }
}
