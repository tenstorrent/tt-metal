// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_xffn_compute.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
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
// PMK_PART_FID (Stage 16 M-B: program B as a window of the fused layer program, whose kernel fidelity is the kda
// items' HiFi4): the matmuls at B's own fidelity (kernels/pmk/pmk_fid.hpp); standalone = the API calls unchanged.
// Selected HERE (this header is included per phase namespace, inside the sub-part's define scope)
// PMK_FFN_FID (Stage 14 B2.3: the FFN's LoFi under the combine's kernel fidelity -- the combine's bf16 x weight
// multiplies at LoFi read 0.981 x HiFi2's y norm, d0 b23w 2026-10-25; bfp4 x bf16 matmuls are bitwise LoFi = HiFi2)
// wins over PMK_PART_FID
#undef FFN_MM_INIT
#undef FFN_MM
#if defined(PMK_FFN_FID)
#define FFN_MM_INIT(...) pmk::mm_block_init_fid<PMK_FFN_FID>(__VA_ARGS__)
#define FFN_MM(...) pmk::mm_block_fid<PMK_FFN_FID>(__VA_ARGS__)
#elif defined(PMK_PART_FID)
#define FFN_MM_INIT(...) pmk::mm_block_init_fid<PMK_PART_FID>(__VA_ARGS__)
#define FFN_MM(...) pmk::mm_block_fid<PMK_PART_FID>(__VA_ARGS__)
#else
#define FFN_MM_INIT(...) matmul_block_init(__VA_ARGS__)
#define FFN_MM(...) matmul_block(__VA_ARGS__)
#endif
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"

using namespace pmk_moe;

// the CB holding K chunk kc's x rows: CB_XRM, or with XFFN_X_SPLIT CB_XRM (even chunks, reader) / CB_XRM1 (odd, writer)
FORCE_INLINE uint32_t xrm_cb(uint32_t kc) {
#ifdef XFFN_X_SPLIT
    return (kc & 1) ? CB_XRM1 : CB_XRM;
#else
    (void)kc;
    return CB_XRM;
#endif
}

FORCE_INLINE void phase_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t W_T = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t KC = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t NC = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t MG = get_compile_time_arg_val(CT0 + 4);
    constexpr uint32_t SUBW = get_compile_time_arg_val(CT0 + 5);
    constexpr uint32_t SUBA = get_compile_time_arg_val(CT0 + 6);
    constexpr uint32_t GU_T = 2 * W_T;
    static_assert(GU_T % SUBA == 0 && NC % SUBW == 0 && SUBW <= 4 && SUBA <= 8, "fp32 DEST subblocks");

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_XT, CB_WGU, CB_GUP);
    while (true) {
        cb_wait_front(CB_CTL, 1);
        const uint32_t ctl = read_tile_value(CB_CTL, 0, 0);
        const uint32_t mt = ctl & 0xFFu;
#ifdef PMK_GU_SPLIT
        const uint32_t spl_c0 = read_tile_value(CB_CTL, 0, 1), spl_cw = read_tile_value(CB_CTL, 0, 2);
#endif
        cb_pop_front(CB_CTL, 1);
        if (ctl == 0) {
            break;
        }
#ifdef PMK_GU_SPLIT
        {
            // perf5 PMK_GU_SPLIT (moe2/PERF.md sec 12): gate|up for columns [c0, c0 + cw) of the stream's mt = R' row
            // tiles (every member of the rectangle is busy: the per-member tile-mults per chunk fall from 256 to 256 R'
            // / g), the activation of this slice's pairs into CB_ACTS (the NCRISC sends it to the row owners), then the
            // down phase on the OWNED row only (CB_ACT = act[own row, every column], landed over the NoC and pushed by
            // the BRISC once SEM_ACT says all g senders are in). Each output tile keeps its K order: bitwise vs the row
            // form.
#ifdef XFFN_X_TILED
#error "PMK_GU_SPLIT: the DRAM x gather path only (x_l1 0)"
#endif
            constexpr uint32_t GUPT = PMK_GU_SPLIT_GUPT, ACTST = GUPT / 2, XTT = PMK_GU_SPLIT_RMAX * KC;
            static_assert(PMK_GU_SPLIT_SUBA == SUBA, "the column slices are dealt in SUBA-tile units");
            cb_reserve_back(CB_ACTS, ACTST);  // the previous stream's slices left (the NCRISC popped them)
            cb_reserve_back(CB_GUP, GUPT);
            for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
                const uint32_t cbx = xrm_cb(kc);
                cb_reserve_back(CB_XT, XTT);
                reconfig_data_format_srca(cbx);
                pack_reconfig_data_format(CB_XT);
                tilize_init(cbx, KC, CB_XT);
                for (uint32_t r = 0; r < mt; ++r) {  // the writer pushes KC pages per row tile of the stream
                    cb_wait_front(cbx, KC);
                    tilize_block(cbx, KC, CB_XT, 0, r * KC);
                    cb_pop_front(cbx, KC);
                }
                tilize_uninit(cbx, CB_XT);
                cb_push_back(CB_XT, XTT);
                cb_wait_front(CB_XT, XTT);
                cb_wait_front(CB_WGU, KC * GU_T);
                if (spl_cw) {
                    reconfig_data_format(CB_WGU, CB_XT);
                    pack_reconfig_data_format(CB_GUP);
                    FFN_MM_INIT(CB_XT, CB_WGU, false, SUBA, 1, KC);
                    PACK((llk_pack_reconfig_l1_acc(kc > 0 ? 1 : 0)));
                    for (uint32_t r = 0; r < mt; ++r) {
                        for (uint32_t c = 0; c < spl_cw; c += SUBA) {
                            tile_regs_acquire();
                            for (uint32_t k = 0; k < KC; ++k) {
                                FFN_MM(CB_XT, CB_WGU, r * KC + k, k * GU_T + spl_c0 + c, 0, false, SUBA, 1, KC);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t s = 0; s < SUBA; ++s) {
                                pack_tile<true>(s, CB_GUP, r * spl_cw + c + s);
                            }
                            tile_regs_release();
                        }
                    }
                    PACK((llk_pack_reconfig_l1_acc(0)));
                }
                cb_pop_front(CB_WGU, KC * GU_T);
                cb_pop_front(CB_XT, XTT);
            }
            cb_push_back(CB_GUP, GUPT);
            // act: this slice's (gate_j, up_j) pairs of every stream row -> CB_ACTS tile (r, j)
            cb_wait_front(CB_GUP, GUPT);
            if (spl_cw) {
                reconfig_data_format_srca(CB_GUP);
                pack_reconfig_data_format(CB_ACTS);
                copy_init(CB_GUP);
                clamped_silu_glu_tile_init();
                for (uint32_t r = 0; r < mt; ++r) {
                    for (uint32_t j = 0; j < spl_cw / 2; ++j) {
                        tile_regs_acquire();
                        copy_tile(CB_GUP, r * spl_cw + 2 * j, 0);
                        copy_tile(CB_GUP, r * spl_cw + 2 * j + 1, 1);
                        clamped_silu_glu_tile(0, 1, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(0, CB_ACTS, r * (spl_cw / 2) + j);
                        tile_regs_release();
                    }
                }
            }
            cb_push_back(CB_ACTS, ACTST);
            cb_pop_front(CB_GUP, GUPT);
            // B: down on the owned row (act[row, all W_T columns] from the g senders)
            if (ctl & CTL_NOROW) {
                for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
                    cb_wait_front(CB_WD, W_T * NC);
                    cb_pop_front(CB_WD, W_T * NC);
                }
                continue;
            }
            cb_wait_front(CB_ACT, W_T);
            reconfig_data_format(CB_WD, CB_ACT);
            pack_reconfig_data_format(CB_YRM);
#ifndef PMK_OUT_BFP8
            pack_untilize_dest_init<SUBW, NC>(CB_YRM);
#endif
            FFN_MM_INIT(CB_ACT, CB_WD, false, SUBW, 1, W_T);
            for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
                cb_wait_front(CB_WD, W_T * NC);
                cb_reserve_back(CB_YRM, NC);
                for (uint32_t sb = 0; sb < NC / SUBW; ++sb) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < W_T; ++k) {
                        FFN_MM(CB_ACT, CB_WD, k, k * NC + sb * SUBW, 0, false, SUBW, 1, W_T);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
#ifdef PMK_OUT_BFP8
                    for (uint32_t t = 0; t < SUBW; ++t) {
                        pack_tile(t, CB_YRM, sb * SUBW + t);
                    }
#else
                    pack_untilize_dest<SUBW, NC>(CB_YRM, 1, sb);
#endif
                    tile_regs_release();
                }
                cb_push_back(CB_YRM, NC);
                cb_pop_front(CB_WD, W_T * NC);
            }
#ifndef PMK_OUT_BFP8
            pack_untilize_uninit(CB_YRM);
#endif
            cb_pop_front(CB_ACT, W_T);
            continue;
        }
#endif
        if (ctl & CTL_SKIP) {  // PMK_MCAST: consume the rectangle's stream without math (ring positions stay aligned)
            for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
                cb_wait_front(CB_WGU, KC * GU_T);
                cb_pop_front(CB_WGU, KC * GU_T);
            }
#ifndef XFFN_NO_DOWN
            for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
                cb_wait_front(CB_WD, W_T * NC);
                cb_pop_front(CB_WD, W_T * NC);
            }
#endif
            continue;
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
#ifdef XFFN_X_TILED
            // lever 2: the writer gathered the chunk's x tiles straight into CB_XT (TILE): no tilize
            cb_wait_front(CB_XT, MG * KC);
#else
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
#endif
            cb_wait_front(CB_WGU, KC * GU_T);
            reconfig_data_format(CB_WGU, CB_XT);
            pack_reconfig_data_format(CB_GUP);
#ifdef PMK_GU_RT
            // PERF.md knob: ALL mt rows of the group in dest per subblock (rt = mt, ct = PMK_GU_CT tiles, mt x ct <=
            // PMK_DEST_TILES). The LLK reuses operand A across the columns when ct >= rt and operand B (the WEIGHT
            // tile) across the ROWS when ct < rt (llk_unpack_AB_matmul: reuse_a = ct_dim >= rt_dim), so ct = 1 at
            // MG = 2 / ct = 2 at MG = 4 unpacks every weight tile ONCE per K step instead of once per row. Each dest
            // tile still accumulates its K in order and is packed with the same L1 accumulation: bitwise vs below.
            constexpr uint32_t GU_CT = PMK_GU_CT;
            static_assert(GU_T % GU_CT == 0 && MG * GU_CT <= PMK_DEST_TILES, "rows x columns of a subblock in dest");
            FFN_MM_INIT(CB_XT, CB_WGU, false, GU_CT, mt, KC);
            PACK((llk_pack_reconfig_l1_acc(kc > 0 ? 1 : 0)));
            for (uint32_t nb = 0; nb < GU_T / GU_CT; ++nb) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < KC; ++k) {
                    FFN_MM(CB_XT, CB_WGU, k, k * GU_T + nb * GU_CT, 0, false, GU_CT, mt, KC);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < mt; ++r) {
                    for (uint32_t s = 0; s < GU_CT; ++s) {
                        pack_tile<true>(r * GU_CT + s, CB_GUP, r * GU_T + nb * GU_CT + s);
                    }
                }
                tile_regs_release();
            }
#else
            FFN_MM_INIT(CB_XT, CB_WGU, false, SUBA, 1, KC);
            PACK((llk_pack_reconfig_l1_acc(kc > 0 ? 1 : 0)));
            for (uint32_t r = 0; r < mt; ++r) {
                for (uint32_t nb = 0; nb < GU_T / SUBA; ++nb) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < KC; ++k) {
                        FFN_MM(CB_XT, CB_WGU, r * KC + k, k * GU_T + nb * SUBA, 0, false, SUBA, 1, KC);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t s = 0; s < SUBA; ++s) {
                        pack_tile<true>(s, CB_GUP, r * GU_T + nb * SUBA + s);
                    }
                    tile_regs_release();
                }
            }
#endif
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
        // ---- B: down, untilized straight into the tile row's ROW_MAJOR block (PMK_OUT_BFP8: packed as bfp8 TILES
        // instead -- CB_YRM is then NC bfp8 tiles per tile row and N chunk; the writer re-lays them per row)
        cb_wait_front(CB_ACT, MG * W_T);
        reconfig_data_format(CB_WD, CB_ACT);
        pack_reconfig_data_format(CB_YRM);
#ifndef PMK_OUT_BFP8
        pack_untilize_dest_init<SUBW, NC>(CB_YRM);
#endif
        FFN_MM_INIT(CB_ACT, CB_WD, false, SUBW, 1, W_T);
        for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
            cb_wait_front(CB_WD, W_T * NC);
            for (uint32_t r = 0; r < mt; ++r) {
                cb_reserve_back(CB_YRM, NC);  // NC tile-sized pages = the tile row's 32 rows x NC tiles ROW_MAJOR
                for (uint32_t sb = 0; sb < NC / SUBW; ++sb) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < W_T; ++k) {
                        FFN_MM(CB_ACT, CB_WD, r * W_T + k, k * NC + sb * SUBW, 0, false, SUBW, 1, W_T);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
#ifdef PMK_OUT_BFP8
                    for (uint32_t t = 0; t < SUBW; ++t) {
                        pack_tile(t, CB_YRM, sb * SUBW + t);
                    }
#else
                    pack_untilize_dest<SUBW, NC>(CB_YRM, 1, sb);
#endif
                    tile_regs_release();
                }
                cb_push_back(CB_YRM, NC);
            }
            cb_pop_front(CB_WD, W_T * NC);
        }
#ifndef PMK_OUT_BFP8
        pack_untilize_uninit(CB_YRM);
#endif
        cb_pop_front(CB_ACT, MG * W_T);
    }
}
