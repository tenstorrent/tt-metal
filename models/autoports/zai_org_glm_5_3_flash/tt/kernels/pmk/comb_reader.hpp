// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_combine_rm_reader.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// Stage 16 combine "rm" (tt/prefill_tp.py EpMoe._combine_rm, GLM53_PF_MOE_COMBINE=rm) -- reader: the gather of the
// weighted top-k sum y[t] = sum_j w[t, j] * out[dest[j C + t]], fused (no [C k, H] gathered copy, no separate multiply
// / reduce passes). out = the expert outputs untilized to ROW_MAJOR [CAP, H] bf16 (one page per dispatch row).
//
// Unit u = (token chunk u / H_SPLIT of 32 tokens, column part u % H_SPLIT of PART_BYTES). Per token t of the chunk the
// K rows' parts land in CB_CIN (K x PT pages of 2 KB: each 1,024 bf16 of one row, the compute's "row tiles") and K
// weight tiles in CB_CW (fp32, the weight at datum 0: SCALAR broadcast).
// Compile-time: C, K, ROW_BYTES (= H x 2, the out page), PART_BYTES, PT (= PART_BYTES / 2,048), H_SPLIT.
// Runtime: out, dest ([1, K C] uint32, pair p = j C + t), w ([1, 1, C, K] fp32 TILE: the chunk's 32 x K weights are
// ONE 4-KB tile, (t, j) in face 0 (t < 16) or face 2), first unit, unit count. CB_CIDX scratch: K x 32 dest words,
// then the weight tile (both at 64-B multiples: BH DRAM reads land at 64-B aligned L1 offsets).
// ---- v2 (tt/pmk_moe2.py, pmk/moe2/README.md lever 5): PMK_SHARED -- the P13 shared-expert partial (DRAM ROW_MAJOR
// [C, H] bf16, PMK_RT_YS) is the (K + 1)-th term of every token, weight 1.0, LAST in the fp32 sum (summed before the
// RS, as one partial). PMK_WT_BF16 -- the routing weights come from the family's router tail (router_tail/writer.cpp
// RTAIL_WEIGHTS: [C, K] bf16 ROW_MAJOR, 16 B per token row) instead of the op path's [C, K] fp32 TILE: the chunk's 32
// rows (512 B, 64-B aligned) are read once and each bf16 is widened to fp32 bits (<< 16) at datum 0.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"

// perf5 round 3 (PERF.md sec 15): PMK_COMB_SPLIT -- the including namespace sets HALF: 0 = the even slots, the y_s term
// and the y write (NCRISC; PMK_RT_Y = the y address), 1 = the odd slots (BRISC: CB_CIN2 / CB_CW2, staging in CB_WGU's
// bytes, idle in the combine). XFFN_PROBE: the combine's cycles -- slot 8 gathers (read barriers), 9 the tile rebuild,
// 10 the y_s read / y write, 11 the combine span, 12 its start -- re-flush the RISC's probe row at the end.
FORCE_INLINE void phase_main() {
#ifdef PMK_COMB_SPLIT
    constexpr uint32_t COMB_HALF = HALF, COMB_STEP = 2;
#else
    constexpr uint32_t COMB_HALF = 0, COMB_STEP = 1;
#endif
    constexpr uint32_t CBIN = COMB_HALF ? CB_CIN2 : CB_CIN, CBW = COMB_HALF ? CB_CW2 : CB_CW;
#ifdef XFFN_PROBE
    uint64_t cp_t0 = 0;
    uint32_t cp_gather = 0, cp_rebuild = 0, cp_y = 0;
    const uint64_t cp_start = get_timestamp();
#define CP_TIC() cp_t0 = get_timestamp()
#define CP_TOC(acc) acc += static_cast<uint32_t>(get_timestamp() - cp_t0)
#else
#define CP_TIC() \
    do {         \
    } while (0)
#define CP_TOC(acc) \
    do {            \
    } while (0)
#endif
    constexpr uint32_t C = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t K = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t PART_BYTES = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t PT = get_compile_time_arg_val(CT0 + 4);
    constexpr uint32_t H_SPLIT = get_compile_time_arg_val(CT0 + 5);
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t W_TILE_BYTES = 4096;  // fp32
    static_assert(K <= 16, "the K weights of a token sit in one face row");
    static_assert(PT * TILE_BYTES == PART_BYTES, "a part = whole 1,024-datum row tiles");
    static_assert((C * 4) % 64 == 0, "dest rows of the K slots start 64-B aligned");
#ifndef PMK_COMB_GROUP
#define PMK_COMB_GROUP 1
#endif
    constexpr uint32_t CG = PMK_COMB_GROUP;  // Stage 14 B2.3: tokens per read barrier in the bf16 gather
    static_assert(32 % CG == 0, "whole token groups per 32-token unit");
#if PMK_COMB_GROUP > 1 && (defined(PMK_OUT_BFP8) || defined(PMK_COMB_SPLIT))
#error "PMK_COMB_GROUP: the bf16-row gather only"
#endif
#if defined(PMK_OUT_TM) && (defined(PMK_OUT_BFP8) || defined(PMK_COMB_SPLIT) || defined(PMK_RM_SKIP))
#error "PMK_OUT_TM: the bf16-row gather without RM_SKIP only"
#endif
#ifdef PMK_SHARED
    constexpr uint32_t KK = K + 1;
    const uint32_t ys_addr = get_arg_val<uint32_t>(PMK_RT_YS);
    const InterleavedAddrGen<true> ys_g{.bank_base_address = ys_addr, .page_size = ROW_BYTES};
#else
    constexpr uint32_t KK = K;
#endif
#ifdef PMK_COMB_SLOT
    constexpr uint32_t CIN_SLOT = PMK_COMB_SLOT;  // a token's CIN pages, padded (comb_compute.hpp pops the padding)
#else
    constexpr uint32_t CIN_SLOT = KK * PT;
#endif
    static_assert(CIN_SLOT >= KK * PT, "a token slot holds its KK terms");

    const uint32_t out_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t dest_addr = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t wt_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t u0 = get_arg_val<uint32_t>(RT0 + 3);
    const uint32_t nu = get_arg_val<uint32_t>(RT0 + 4);
    if (nu == 0) {
        return;
    }

    // the weight tiles: datum 0 is written per pair; the rest stays 0 (zeroed once: every page of the CB). PMK_CW_PAGES
    // = the CB's real capacity (its alias group's pages, host): PMK_OUT_BFP8 plans CW at 2 pages, so 2 x KK reserved
    // more than the CB holds at TP8's member blocks (1, 4, 4) (9 pages: the reserve never returns; 16, 2026-10-22
    // review)
#if defined(PMK_COMB_SPLIT) && defined(PMK_CW2_PAGES)
    constexpr uint32_t CW_ZERO = COMB_HALF ? PMK_CW2_PAGES : PMK_CW_PAGES;
#elif defined(PMK_CW_PAGES)
    constexpr uint32_t CW_ZERO = PMK_CW_PAGES;
#else
    constexpr uint32_t CW_ZERO = 2 * KK;
#endif
    {
        cb_reserve_back(CBW, CW_ZERO);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CBW));
        for (uint32_t i = 0; i < CW_ZERO * W_TILE_BYTES / 4; ++i) {
            z[i] = 0;
        }
    }

    const uint32_t tab_ = get_write_ptr(CB_TAB);  // the push probe words
    // CB_CIDX shares its allocation with CB_TAB, whose first SYNC_WORDS_B bytes the RISCs keep writing during the
    // combine (the multicast sync words and the push probe: push_mark below pins CB_TAB + W_PUSH_NCRISC = 80 before
    // every CIN push): the dest words start past them (16, 2026-10-22: at + 0, word 20 = slot 0 of token 20 was
    // overwritten by the probe inside the token loop -> token 20 of every chunk gathered a garbage row: inf)
    // (PMK_COMB_SPLIT half 1: the BRISC's index words / weights / staging live in CB_WGU's bytes, idle in the combine)
    const uint32_t idx_l1 = COMB_HALF ? get_write_ptr(CB_WGU) : get_write_ptr(CB_CIDX) + SYNC_WORDS_B;
    const uint32_t w_l1 = idx_l1 + K * 32 * 4;
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint32_t* wl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> dest_g{.bank_base_address = dest_addr, .page_size = K * C * 4};
#ifdef PMK_WT_BF16
    const InterleavedAddrGen<true> wt_g{.bank_base_address = wt_addr, .page_size = C * 2 * K};  // one ROW_MAJOR page
#else
    const InterleavedAddrGen<true> wt_g{.bank_base_address = wt_addr, .page_size = W_TILE_BYTES};
#endif

#ifdef PMK_OUT_BFP8
    // row-bfp8 partials (moe_common.hpp): per unit (32-token chunk, column part of NT tiles), per slot j: the 32 token
    // rows' parts (NT / NC chunks of RB bytes each, 64-B reads) land in the staging area past the index words, then
    // the RISC rebuilds NT bfp8 tiles (token t = tile row t: exponent byte (t / 16) * 2 + ... , 16-B mantissa pieces)
    // into CB_CIN (NT pages of 1,088 B) and the weights of the 32 tokens for slot j into column 0 of one fp32 weight
    // tile (CB_CW: COL broadcast in the compute). With PMK_SHARED the P13 partial y_s (bf16 TILE [C, H]: the PfStream
    // pair's native output) is the LAST term: its NT tiles go through CB_CSH (bf16) with a ones weight column.
    constexpr uint32_t NC_ = PMK_OUT_NC;
    constexpr uint32_t RB = ROWB_EXP_PAD + 32 * NC_;
    constexpr uint32_t H_T_ = ROW_BYTES / 64;
    constexpr uint32_t ROWB_BYTES = (H_T_ / NC_) * RB;
    constexpr uint32_t NT = H_T_ / H_SPLIT;  // tiles of a part
    static_assert(NT % NC_ == 0, "a part = whole N chunks");
    constexpr uint32_t PART_B = (NT / NC_) * RB;
    const InterleavedAddrGen<true> outb_g{.bank_base_address = out_addr, .page_size = ROWB_BYTES};
    const uint32_t stage = (w_l1 + W_TILE_BYTES + 63) & ~63u;  // 32 x PART_B staging (CB_CIDX sized by the host)
#endif
#ifdef PMK_RM_SKIP
    // W7 step 5 (G; 14's GLM53_PF_MOE_RM_SKIP, W8 unit 1): a pair of another chip's expert (dest = the PMK_EP_SENTINEL
    // of plan_dest.hpp) or of weight +-0 reads no DRAM row -- its slot is a local copy of one zeroed part (CB_CZERO),
    // so 0 x 0 adds the +0 the zero row added; no unwritten out_rm row is ever read
#if defined(PMK_OUT_BFP8) || defined(PMK_COMB_SPLIT)
#error "PMK_RM_SKIP: the bf16-row combine only"
#endif
    const uint32_t zero_l1 = get_write_ptr(CB_CZERO);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
        for (uint32_t i = 0; i < PART_BYTES / 4; ++i) {
            z[i] = 0;
        }
    }
    const uint64_t zero_noc = get_noc_addr(zero_l1);
#endif
#ifdef PMK_OUT_TM
    // Stage 14 B2.3 token-major out_rm (ffn_writer.hpp PMK_OUT_TM): page (part, token) = the token's K row-parts in
    // slot order, contiguous: ONE read per token-part lands the K CIN pages (no dest table)
    static_assert(PMK_TM_PART == PART_BYTES && PMK_TM_K == K, "the writer's page geometry");
    const InterleavedAddrGen<true> tm_g{.bank_base_address = out_addr, .page_size = K * PART_BYTES};
#endif
#if !defined(PMK_COMB_STRIDE)
#define PMK_COMB_STRIDE \
    1  // units u0, u0 + 1, .. (contiguous); PMK_COMB_STRIDE = P: u0 + k P (the block halves land in rounds)
#endif
    for (uint32_t k_u = 0; k_u < nu; ++k_u) {
#if defined(PMK_COMB_UNITS)  // a per-rank start (the half this chip SENDS first in a streamed P15) wraps around
        const uint32_t u = (u0 + k_u * PMK_COMB_STRIDE) % PMK_COMB_UNITS;
#else
        const uint32_t u = u0 + k_u * PMK_COMB_STRIDE;
#endif
        const uint32_t chunk = u / H_SPLIT;
        const uint32_t part = u - chunk * H_SPLIT;
        const uint32_t t0 = chunk * 32;
#ifndef PMK_OUT_TM
        for (uint32_t j = COMB_HALF; j < K; j += COMB_STEP) {
            noc_async_read(dest_g.get_noc_addr(0, (j * C + t0) * 4), idx_l1 + j * 128, 128);
        }
#endif
#ifdef PMK_OUT_BFP8
#ifdef PMK_WT_BF16
        noc_async_read(wt_g.get_noc_addr(0, t0 * 2 * K), w_l1, 32 * 2 * K);
#else
        noc_async_read(wt_g.get_noc_addr(chunk), w_l1, W_TILE_BYTES);
#endif
        noc_async_read_barrier();
        for (uint32_t j = COMB_HALF; j < K; j += COMB_STEP) {
            CP_TIC();
            for (uint32_t t = 0; t < 32; ++t) {
                noc_async_read(outb_g.get_noc_addr(idx[j * 32 + t], part * PART_B), stage + t * PART_B, PART_B);
            }
            cb_reserve_back(CBIN, NT);
            cb_reserve_back(CBW, 1);
            const uint32_t cin = get_write_ptr(CBIN);
            const uint32_t ww = get_write_ptr(CBW);
            noc_async_read_barrier();
            CP_TOC(cp_gather);
            CP_TIC();
            for (uint32_t n = 0; n < NT; ++n) {  // rebuild tile n of the part from the 32 rows
                const uint32_t tile = cin + n * BFP8_TILE_B;
                const uint32_t cn = n / NC_, within = n - cn * NC_;
                volatile tt_l1_ptr uint8_t* te = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile);
                for (uint32_t t = 0; t < 32; ++t) {
                    const uint32_t src = stage + t * PART_B + cn * RB;
                    const uint32_t f = (t >> 4) * 2, rr = t & 15u;
                    volatile tt_l1_ptr uint8_t* se = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
                    te[f * 16 + rr] = se[2 * within];
                    te[(f + 1) * 16 + rr] = se[2 * within + 1];
                    volatile tt_l1_ptr uint32_t* sm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + ROWB_EXP_PAD + within * 32);
                    volatile tt_l1_ptr uint32_t* d0_ =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + BFP8_EXP_B + f * 256 + rr * 16);
                    volatile tt_l1_ptr uint32_t* d1_ = d0_ + 64;
                    for (uint32_t w_ = 0; w_ < 4; ++w_) {
                        d0_[w_] = sm[w_];
                        d1_[w_] = sm[4 + w_];
                    }
                }
            }
            // the weight tile: column 0 = the 32 tokens' weights of slot j (fp32 tile: face (t / 16) * 2, row t % 16)
            for (uint32_t t = 0; t < 32; ++t) {
                const uint32_t off = ((t >> 4) * 2) * 1024 + (t & 15u) * 64;
#ifdef PMK_WT_BF16
                const uint32_t bits =
                    static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1)[t * K + j]) << 16;
#else
                const uint32_t bits = wl[(t < 16 ? t * 16 : 512 + (t - 16) * 16) + j];
#endif
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ww + off) = bits;
            }
            CP_TOC(cp_rebuild);
            push_mark(tab_, COMB_HALF ? W_PUSH_BRISC : W_PUSH_NCRISC, CBIN, NT);
            cb_push_back(CBIN, NT);
            cb_push_back(CBW, 1);
        }
#ifdef PMK_SHARED
        if constexpr (COMB_HALF == 0) {  // the P13 partial, LAST: the chunk's NT tiles of y_s (bf16 TILE [C, H]: tile
                                         // (chunk, n) = page chunk H_T + n)
            // into CB_CSH and a weight tile whose column 0 is 1.0
            const InterleavedAddrGen<true> yst_g{.bank_base_address = ys_addr, .page_size = 2048};
            cb_reserve_back(CB_CSH, NT);
            cb_reserve_back(CB_CW, 1);
            const uint32_t csh = get_write_ptr(CB_CSH);
            for (uint32_t n = 0; n < NT; ++n) {
                noc_async_read(yst_g.get_noc_addr(chunk * H_T_ + part * NT + n), csh + n * 2048, 2048);
            }
            const uint32_t ww = get_write_ptr(CB_CW);
            for (uint32_t t = 0; t < 32; ++t) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ww + ((t >> 4) * 2) * 1024 + (t & 15u) * 64) =
                    0x3F800000u;
            }
            CP_TIC();
            noc_async_read_barrier();
            CP_TOC(cp_y);
            push_mark(tab_, W_PUSH_NCRISC, CB_CSH, NT);
            cb_push_back(CB_CSH, NT);
            cb_push_back(CB_CW, 1);
        }
#endif
#if defined(PMK_COMB_SPLIT) && defined(PMK_RT_Y)  // (PMK_RT_Y: the NCRISC's instance only)
        if constexpr (COMB_HALF == 0) {  // the y write (comb_writer.hpp's unit) runs here: the BRISC rebuilds the odd
                                         // slots
            const InterleavedAddrGen<true> yg{.bank_base_address = get_arg_val<uint32_t>(PMK_RT_Y), .page_size = 2048};
            const uint32_t base = chunk * H_T_ + part * NT;
            cb_wait_front(CB_COUT, NT);
            CP_TIC();
            const uint32_t rp = get_read_ptr(CB_COUT);
            for (uint32_t x = 0; x < NT; ++x) {
                noc_async_write(rp + x * 2048, yg.get_noc_addr(base + x), 2048);
            }
            noc_async_write_barrier();
            CP_TOC(cp_y);
            cb_pop_front(CB_COUT, NT);
        }
#endif
        continue;
#endif
#ifdef PMK_WT_BF16
        noc_async_read(wt_g.get_noc_addr(0, t0 * 2 * K), w_l1, 32 * 2 * K);  // the chunk's 32 rows x K bf16
#else
        noc_async_read(wt_g.get_noc_addr(chunk), w_l1, W_TILE_BYTES);
#endif
        noc_async_read_barrier();
        // Stage 14 B2.3 PMK_COMB_GROUP = CG tokens per read barrier (default 1): the gather was latency-bound (one
        // token's 9 x 2-KB reads in flight, then a full barrier: 150 MB at ~200 GB/s, d0 bisect 2026-10-25 23:46Z);
        // CG tokens' reads go out together, then one barrier and one push. Token slots of CIN_SLOT pages
        // (PMK_COMB_SLOT: padded so that CG slots tile the idle XRM alias group); the compute pops the padding. Same
        // pages, same order.
        for (uint32_t tg = 0; tg < 32; tg += CG) {
            cb_reserve_back(CB_CIN, CG * CIN_SLOT);
            const uint32_t wp0 = get_write_ptr(CB_CIN);
            cb_reserve_back(CB_CW, CG * KK);
            const uint32_t ww0 = get_write_ptr(CB_CW);
            for (uint32_t g_ = 0; g_ < CG; ++g_) {
                const uint32_t t = tg + g_;
                const uint32_t wp = wp0 + g_ * CIN_SLOT * TILE_BYTES;
#ifdef PMK_OUT_TM
                noc_async_read(tm_g.get_noc_addr(part * C + t0 + t), wp, K * PART_BYTES);
#else
                for (uint32_t j = 0; j < K; ++j) {
#ifdef PMK_RM_SKIP
#ifdef PMK_WT_BF16
                    const bool w0 = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1)[t * K + j] & 0x7fffu) == 0;
#else
                    const bool w0 = (wl[(t < 16 ? t * 16 : 512 + (t - 16) * 16) + j] & 0x7fffffffu) == 0;
#endif
                    if (idx[j * 32 + t] == 0xFFFFFFFFu || w0) {
                        noc_async_read(zero_noc, wp + j * PART_BYTES, PART_BYTES);
                        continue;
                    }
#endif
                    noc_async_read(
                        out_g.get_noc_addr(idx[j * 32 + t], part * PART_BYTES), wp + j * PART_BYTES, PART_BYTES);
                }
#endif
#ifdef PMK_SHARED
                noc_async_read(ys_g.get_noc_addr(t0 + t, part * PART_BYTES), wp + K * PART_BYTES, PART_BYTES);
#endif
                const uint32_t ww = ww0 + g_ * KK * W_TILE_BYTES;
#ifdef PMK_WT_BF16
                volatile tt_l1_ptr uint16_t* wb = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1);
                for (uint32_t j = 0; j < K; ++j) {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ww + j * W_TILE_BYTES) =
                        static_cast<uint32_t>(wb[t * K + j]) << 16;
                }
#else
                const uint32_t wrow = t < 16 ? t * 16 : 512 + (t - 16) * 16;  // face 0 / face 2, row t % 16
                for (uint32_t j = 0; j < K; ++j) {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ww + j * W_TILE_BYTES) = wl[wrow + j];
                }
#endif
#ifdef PMK_SHARED
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ww + K * W_TILE_BYTES) = 0x3F800000u;  // 1.0f
#endif
            }
            noc_async_read_barrier();
            push_mark(tab_, W_PUSH_NCRISC, CB_CIN, CG * CIN_SLOT);
            cb_push_back(CB_CIN, CG * CIN_SLOT);
            cb_push_back(CB_CW, CG * KK);
        }
    }
#ifdef XFFN_PROBE
    {  // the combine's cycles into this RISC's probe row (slots 8-12), re-flushed with the FFN counters
        xffn_probe::g_c[8] = cp_gather;
        xffn_probe::g_c[9] = cp_rebuild;
        xffn_probe::g_c[10] = cp_y;
        xffn_probe::g_c[11] = static_cast<uint32_t>(get_timestamp() - cp_start);
        xffn_probe::g_c[12] = static_cast<uint32_t>(cp_start);
        if constexpr (COMB_HALF == 0) {
            XP_FLUSH(
                (get_write_ptr(CB_WTAB) + 63) & ~63u,
                get_arg_val<uint32_t>(PMK_RT_FFN + 4),
                get_arg_val<uint32_t>(PMK_RT_FFN + 5) + get_arg_val<uint32_t>(PMK_RT_FFN + 6));
        } else {
            XP_FLUSH(
                (get_write_ptr(CB_TAB) + SYNC_WORDS_B + 63) & ~63u,
                get_arg_val<uint32_t>(PMK_RT_FFN + 7),
                get_arg_val<uint32_t>(PMK_RT_FFN + 5));
        }
    }
#endif
}
