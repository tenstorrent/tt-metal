// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_combine_rm_compute.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the
// ORIGINAL kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at
// RT0 + i (the including composite defines both before this header, inside the phase's namespace) and its CBs at
// pmk_moe's indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the
// original, regenerate.
// ---- original header follows ----
// Stage 16 combine "rm" -- compute (moe_combine_rm_reader.cpp): per token, the K gathered row parts x their weights
// (SCALAR broadcast) accumulated in fp32 in L1 (packer L1 accumulation into CB_CRM, row t at pages t PT ..), then the
// chunk's 32 rows tilized into CB_COUT (32 PT bf16 tiles). Compile-time: K, PT. Runtime: unit count.
// fp32_dest_acc_en: a row part (PT <= 4 tiles) fits the half-sync DEST.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
// PMK_PART_FID (Stage 16 M-B, see ffn_compute.hpp): the broadcast multiplies at B's own fidelity in the fused program
#undef COMB_MUL_COLS_INIT
#undef COMB_MUL_COLS
#undef COMB_MUL_SCALAR_INIT
#undef COMB_MUL_SCALAR
#if defined(PMK_PART_FID)
#define COMB_MUL_COLS_INIT(a, b) pmk::mul_bcast_init_fid<PMK_PART_FID, BroadcastType::COL>(a, b)
#define COMB_MUL_COLS(...) pmk::mul_tiles_bcast_fid<PMK_PART_FID, BroadcastType::COL>(__VA_ARGS__)
#define COMB_MUL_SCALAR_INIT(a, b) pmk::mul_bcast_init_fid<PMK_PART_FID, BroadcastType::SCALAR>(a, b)
#define COMB_MUL_SCALAR(...) pmk::mul_tiles_bcast_fid<PMK_PART_FID, BroadcastType::SCALAR>(__VA_ARGS__)
#else
#define COMB_MUL_COLS_INIT(a, b) mul_bcast_cols_init_short(a, b)
#define COMB_MUL_COLS(...) mul_tiles_bcast<BroadcastType::COL>(__VA_ARGS__)
#define COMB_MUL_SCALAR_INIT(a, b) mul_bcast_scalar_init(a, b)
#define COMB_MUL_SCALAR(...) mul_tiles_bcast<BroadcastType::SCALAR>(__VA_ARGS__)
#endif

// PMK_COMB_FID (Stage 14, numaudit/combine_fid): the combine's multiplies at ITS OWN fidelity, over PMK_PART_FID / the
// composite's MATH_FIDELITY: the B composite runs the bfp4 experts at HiFi2 while the op's combine is HiFi4 (at HiFi2
// every MoE output shrinks 0.33-0.60 % more: the weight's SrcB bits). The value = ckernel::MathFidelity (4 = HiFi4).
#if defined(PMK_COMB_FID)
#undef COMB_MUL_COLS_INIT
#undef COMB_MUL_COLS
#undef COMB_MUL_SCALAR_INIT
#undef COMB_MUL_SCALAR
#define COMB_MUL_COLS_INIT(a, b) pmk::mul_bcast_init_fid<PMK_COMB_FID, BroadcastType::COL>(a, b)
#define COMB_MUL_COLS(...) pmk::mul_tiles_bcast_fid<PMK_COMB_FID, BroadcastType::COL>(__VA_ARGS__)
#define COMB_MUL_SCALAR_INIT(a, b) pmk::mul_bcast_init_fid<PMK_COMB_FID, BroadcastType::SCALAR>(a, b)
#define COMB_MUL_SCALAR(...) pmk::mul_tiles_bcast_fid<PMK_COMB_FID, BroadcastType::SCALAR>(__VA_ARGS__)
#endif

FORCE_INLINE void phase_main() {
    constexpr uint32_t K = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t PT = get_compile_time_arg_val(CT0 + 1);
    static_assert(PT <= 4, "a row part must fit the fp32 half-sync DEST");
#ifdef PMK_SHARED
    constexpr uint32_t KK = K + 1;  // v2: the shared expert's partial is the last term (weight 1.0)
#else
    constexpr uint32_t KK = K;
#endif
    const uint32_t nu = get_arg_val<uint32_t>(RT0 + 0);
    if (nu == 0) {
        return;
    }
    // Stage 14 B2.3 PMK_COMB_GROUP (comb_reader.hpp): a token's CIN pages may be padded to PMK_COMB_SLOT (the padding
    // pages carry nothing: popped after the token's KK terms)
#ifdef PMK_COMB_SLOT
    constexpr uint32_t CIN_SLOT = PMK_COMB_SLOT;
#else
    constexpr uint32_t CIN_SLOT = KK * PT;
#endif
    static_assert(CIN_SLOT >= KK * PT, "a token slot holds its KK terms");

#if defined(PMK_BW_STAGE) && PMK_BW_STAGE == 3 && !defined(PMK_OUT_BFP8)
    // the M-B1 combine bisect (pmk_b_window GLM53_PMK_B_STAGE=3): the reader only -- every token's CIN / CW pages
    // drained
    for (uint32_t u = 0; u < nu; ++u) {
        for (uint32_t t = 0; t < 32; ++t) {
            cb_wait_front(CB_CIN, CIN_SLOT);
            cb_pop_front(CB_CIN, CIN_SLOT);
            cb_wait_front(CB_CW, KK);
            cb_pop_front(CB_CW, KK);
        }
    }
    return;
#endif
    compute_kernel_hw_startup(CB_CIN, CB_CW, CB_CRM);
#ifdef PMK_OUT_BFP8
    // row-bfp8 partials: per unit, per slot j, NT rebuilt bfp8 tiles x the tokens' weight tile (COL broadcast: row t of
    // every tile x column 0's weight of token t), fp32 L1-accumulated over the K slots in CB_CRM (NT fp32 tiles), then
    // copied to bf16 tiles in CB_COUT (no tilize: the output is already TILE). The fixed slot order is kept; with
    // PMK_SHARED the P13 partial (CB_CSH, bf16 tiles, weight 1.0) is the last term before the copy.
    constexpr uint32_t NT = PMK_OUT_NT;
    for (uint32_t u = 0; u < nu; ++u) {
        reconfig_data_format(CB_CIN, CB_CW);
        pack_reconfig_data_format(CB_CRM);
        COMB_MUL_COLS_INIT(CB_CIN, CB_CW);
        cb_reserve_back(CB_CRM, NT);
        for (uint32_t j = 0; j < K; ++j) {
#ifdef PMK_COMB_SPLIT
            const uint32_t cin = (j & 1) ? CB_CIN2 : CB_CIN, cw = (j & 1) ? CB_CW2 : CB_CW;  // odd slots: the BRISC's
#else
            const uint32_t cin = CB_CIN, cw = CB_CW;
#endif
            cb_wait_front(cin, NT);
            cb_wait_front(cw, 1);
            pack_reconfig_l1_acc(j == 0 ? 0 : 1);
            for (uint32_t n = 0; n < NT; n += 4) {
                tile_regs_acquire();
                for (uint32_t x = 0; x < 4; ++x) {
                    COMB_MUL_COLS(cin, cw, n + x, 0, x);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t x = 0; x < 4; ++x) {
                    pack_tile<true>(x, CB_CRM, n + x);
                }
                tile_regs_release();
            }
            cb_pop_front(cin, NT);
            cb_pop_front(cw, 1);
        }
#ifdef PMK_SHARED
        {  // the P13 partial, last: bf16 tiles x the ones column (srcA reconfigured to bf16 for this term)
            cb_wait_front(CB_CSH, NT);
            cb_wait_front(CB_CW, 1);
            reconfig_data_format(CB_CSH, CB_CW);
            COMB_MUL_COLS_INIT(CB_CSH, CB_CW);
            pack_reconfig_l1_acc(1);
            for (uint32_t n = 0; n < NT; n += 4) {
                tile_regs_acquire();
                for (uint32_t x = 0; x < 4; ++x) {
                    COMB_MUL_COLS(CB_CSH, CB_CW, n + x, 0, x);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t x = 0; x < 4; ++x) {
                    pack_tile<true>(x, CB_CRM, n + x);
                }
                tile_regs_release();
            }
            cb_pop_front(CB_CSH, NT);
            cb_pop_front(CB_CW, 1);
            reconfig_data_format(CB_CIN, CB_CW);
        }
#endif
        pack_reconfig_l1_acc(0);
        cb_push_back(CB_CRM, NT);
        cb_wait_front(CB_CRM, NT);
        reconfig_data_format_srca(CB_CRM);
        pack_reconfig_data_format(CB_COUT);
        copy_tile_init(CB_CRM);
        cb_reserve_back(CB_COUT, NT);
        for (uint32_t n = 0; n < NT; n += 4) {
            tile_regs_acquire();
            for (uint32_t x = 0; x < 4; ++x) {
                copy_tile(CB_CRM, n + x, x);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t x = 0; x < 4; ++x) {
                pack_tile(x, CB_COUT, n + x);
            }
            tile_regs_release();
        }
        cb_push_back(CB_COUT, NT);
        cb_pop_front(CB_CRM, NT);
    }
    return;
#endif
    for (uint32_t u = 0; u < nu; ++u) {
        reconfig_data_format(CB_CIN, CB_CW);
        pack_reconfig_data_format(CB_CRM);
        COMB_MUL_SCALAR_INIT(CB_CIN, CB_CW);
        cb_reserve_back(CB_CRM, 32 * PT);
        for (uint32_t t = 0; t < 32; ++t) {
            for (uint32_t j = 0; j < KK; ++j) {
                cb_wait_front(CB_CIN, PT);
                cb_wait_front(CB_CW, 1);
                pack_reconfig_l1_acc(j == 0 ? 0 : 1);
                tile_regs_acquire();
                for (uint32_t x = 0; x < PT; ++x) {
                    COMB_MUL_SCALAR(CB_CIN, CB_CW, x, 0, x);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t x = 0; x < PT; ++x) {
                    pack_tile<true>(x, CB_CRM, t * PT + x);
                }
                tile_regs_release();
                cb_pop_front(CB_CIN, PT);
                cb_pop_front(CB_CW, 1);
            }
            if constexpr (CIN_SLOT > KK * PT) {  // PMK_COMB_SLOT: the token slot's padding pages
                cb_wait_front(CB_CIN, CIN_SLOT - KK * PT);
                cb_pop_front(CB_CIN, CIN_SLOT - KK * PT);
            }
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(CB_CRM, 32 * PT);
#if defined(PMK_BW_STAGE) && \
    PMK_BW_STAGE == 4  // the combine bisect: + the multiply / L1-acc pack, no tilize (CRM drained)
        cb_wait_front(CB_CRM, 32 * PT);
        cb_pop_front(CB_CRM, 32 * PT);
#elif defined(PMK_COMB_FAST_TILIZE)
        // M-B1 (2026-10-24 mbfu: B's y 1 ULP off standalone B, PCC 0.9999957): the fused member composite is full-sync
        // DEST (A's union), so the helper below picks the REGULAR tilize (32-bit DEST, the packer converts fp32 ->
        // bf16); standalone B (half sync) runs the BH fast tilize (Fp32_enabled cleared: MOVA2D converts into a 16-bit
        // DEST). The helper's fast path, unrolled (its defaults: srcA + pack reconfig, remap configured, wait per
        // block)
        reconfig_data_format_srca(CB_CRM);
        pack_reconfig_data_format(CB_COUT);
        fast_tilize_init(CB_CRM, 32 * PT, CB_COUT);
        cb_wait_front(CB_CRM, 32 * PT);
        cb_reserve_back(CB_COUT, 32 * PT);
        fast_tilize_block(CB_CRM, 32 * PT, CB_COUT);
        cb_push_back(CB_COUT, 32 * PT);
        cb_pop_front(CB_CRM, 32 * PT);
        fast_tilize_uninit(CB_CRM, CB_COUT, 32 * PT);
#else
        // (cf3h-j, coordinator 2026-10-25T17:41:24Z: 14's Fp32Mode::Lossless tilize cbcd904f43d reverted -- the served
        // line's kernels = cf3h-i's; it returns only with a TP8 numerics row vs fp32 incl. the 512 bucket, ruling
        // 2026-10-24T23:00:27Z)
        compute_kernel_lib::tilize<32 * PT, CB_CRM, CB_COUT>(1);
#endif
    }
}
