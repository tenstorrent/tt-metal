// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_xffn_writer.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// Stage 16 XSL expert FFN (moe_xffn_common.hpp) -- writer (NCRISC, NoC1): per M group (control page: tile rows mt,
// valid rows, first dispatch row d0; mt = 0 = end), per N chunk and tile row, the compute's ROW_MAJOR block (32 rows x
// NC tiles) goes to out_rm rows d0 + 32 r + row at column bytes [nc NC 64, +NC 64) -- only the valid rows (the
// padding rows of a region are never read by the combine). Compile-time: H_T, NC (+ KC, MG with XFFN_X_ON_NOC1).
// Runtime: out_rm address (+ x_ext, disp addresses, disp words with XFFN_X_ON_NOC1).
// XFFN_X_ON_NOC1: this kernel (NCRISC, NoC1) also gathers each group's x rows for the gate|up phase (CB_XRM, through
// the plan's disp rows), so the weight stream (reader, NoC0) and the activation gather use different NoCs. With
// XFFN_X_SPLIT only the ODD K chunks (into CB_XRM1; the reader gathers the even ones into CB_XRM): the gather -- the
// critical path of the gate|up phase (in-program probe, 2026-10-22) -- runs on both RISCs / NoCs. XFFN_VALID_ROWS (and
// XFFN_X_SPLIT): only the group's valid rows are gathered (x_ext then needs no zero row).
// XFFN_PROBE: runtime arg 4 = the probe tensor (moe_xffn_probe.hpp), row P + core (P, core: args 5, 6).
// ---- v2 (tt/pmk_moe2.py, pmk/moe2/README.md lever 2) XFFN_X_TILED: the x rows come from the x-hat_f L1 STORE (the
// family's XS 35 = H2 / H8: token row r lives in holder core r / 32's block, bf16 TILE, tile k at base + 2048 k) over
// NoC1 instead of DRAM x_ext, straight into the TILE-layout CB_XT (no tilize on the compute): row t of a 32 x 32 bf16
// tile = 32 B in face (t / 16) * 2 (cols 0..15) + 32 B in the next face (cols 16..31), at (t % 16) * 32 within the
// face; the destination tile (r / 32 of the group, k of the chunk) takes the same two 32-B pieces at the same offsets.
// Runtime (PMK_RT_XS): xs_base, n holders, then (noc x, noc y) per holder. CTL_SKIP (PMK_MCAST): a unit this core only
// receives -- no gather, no output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"

using namespace pmk_moe;
using namespace xffn_probe_slot;

FORCE_INLINE void phase_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t NC = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t ROW_BYTES = H_T * 32 * 2;
    constexpr uint32_t SEG = NC * 32 * 2;
    const uint32_t out_addr = get_arg_val<uint32_t>(RT0 + 0);
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    const uint32_t tab_ = get_write_ptr(CB_TAB);  // the push probe words (moe_common.hpp)
    XP_BEGIN();
#if defined(PMK_XS_ROWS) && (defined(XFFN_X_TILED) || !defined(XFFN_X_ON_NOC1) || defined(XFFN_X_SPLIT) || \
                             !defined(XFFN_VALID_ROWS) || defined(PMK_GU_SPLIT))
#error "PMK_XS_ROWS: the writer's x gather of the valid rows into CB_XRM (x_l1 0 path) only"
#endif
#ifdef PMK_OUT_TM
    // Stage 14 B2.3 PMK_OUT_TM (token-major out_rm, w8/progb_b23_design.md): disp carries the PAIR index p = j C + t
    // (plan_dest.hpp under PMK_OUT_TM); token t = p mod C for the x gather; the y row of pair p goes to page (part q,
    // token t) of [H_SPLIT x C pages of K x PART bytes] at offset j PART + the segment's offset within part q
#if !defined(XFFN_X_ON_NOC1) || defined(XFFN_X_TILED) || defined(XFFN_X_SPLIT) || !defined(XFFN_VALID_ROWS) || \
    defined(PMK_OUT_BFP8) || defined(PMK_GU_SPLIT) || defined(PMK_MCAST)
#error "PMK_OUT_TM: the self-stream bf16 form with the valid-row writer gather only"
#endif
    constexpr uint32_t TM_C = PMK_OUT_TM, TM_PART = PMK_TM_PART, TM_K = PMK_TM_K;
    static_assert(TM_PART % (get_compile_time_arg_val(CT0 + 1) * 64) == 0, "an N chunk's segment sits in one part");
    const InterleavedAddrGen<true> tm_g{.bank_base_address = out_addr, .page_size = TM_K * TM_PART};
#define TM_TOK(v) ((v) % TM_C)
#else
#define TM_TOK(v) (v)
#endif
#if defined(XFFN_X_TILED) || defined(PMK_XS_ROWS)
    // the holders' XS block address: the family passes it (every holder has the same); 0 = this core's own CB_XS
    // (standalone: every core allocates the XS CB, so the address is the holders' too)
    const uint32_t xs_arg = get_arg_val<uint32_t>(PMK_RT_XS + 0);
    const uint32_t xs_base = xs_arg ? xs_arg : get_write_ptr(CB_XS);
    const uint32_t xs_xy = PMK_RT_XS + 2;  // (x, y) of holder h at xs_xy + 2 h
#endif
#ifdef XFFN_X_TILED
    constexpr uint32_t FACE_B = 512;  // 16 x 16 bf16
    constexpr uint32_t PIECE = 32;    // one tile row of one face
#endif
#ifdef XFFN_X_ON_NOC1
    constexpr uint32_t KC = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t MG = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t XSEG = KC * 32 * 2;
    const uint32_t x_addr = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t disp_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t cap = get_arg_val<uint32_t>(RT0 + 3);
    const InterleavedAddrGen<true> x_g{.bank_base_address = x_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = cap * 4};
    volatile tt_l1_ptr uint32_t* dl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WTAB));
#endif

#ifdef PMK_YW_DB
    uint32_t yw_blk = 0;  // groups re-laid so far (scratch = yw_blk mod 2)
#ifndef PMK_YW_COAL
#define PMK_YW_COAL 1
#endif
#endif
    while (true) {
        XP_TIC();
        cb_wait_front(CB_WCTL, 1);
        XP_TOC(W_CTL_WAIT);
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_WCTL));
        const uint32_t ctl = c[0];
#ifdef PMK_GU_SPLIT
        const uint32_t mt = 1;  // the y write covers this member's OWN row only (R' = ctl & 0xFF rows in the stream)
        const uint32_t spl_rr = ctl & 0xFFu, spl_d0 = c[3], spl_rows = c[4], spl_c0 = c[6], spl_cw = c[7];
#else
        const uint32_t mt = ctl & 0xFFu;
#endif
        const uint32_t rows = c[1];
        const uint32_t d0 = c[2];
        cb_pop_front(CB_WCTL, 1);
        if (ctl == 0) {
            break;
        }
#ifdef PMK_GU_SPLIT
        {
            // perf5 PMK_GU_SPLIT: (1) the x rows of ALL R' row tiles of the stream, per row tile (KC pages per push:
            // the compute tilizes row by row), through the stream's disp rows; (2) after the compute's activation of
            // this member's column slice (CB_ACTS, tiles (r, j)), each row r's tiles go to the row owner's CB_ACT (one
            // block: the same L1 address on every member of the rectangle) at column c0 / 2 + j, then +1 on the owner's
            // SEM_ACT behind a write barrier; (3) the y write below is this member's own row (CTL_NOROW: none).
#ifndef XFFN_X_ON_NOC1
#error "PMK_GU_SPLIT needs the writer-side x gather"
#endif
            constexpr uint32_t ACTST = PMK_GU_SPLIT_GUPT / 2;
            XP_TIC();
            noc_async_read(disp_g.get_noc_addr(0, spl_d0 * 4), reinterpret_cast<uint32_t>(dl), spl_rr * 32 * 4);
            noc_async_read_barrier();
            XP_TOC(W_DISP_READ);
            for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
                for (uint32_t r = 0; r < spl_rr; ++r) {
                    XP_TIC();
                    cb_reserve_back(CB_XRM, KC);
                    XP_TOC(W_X_WAIT);
                    XP_TIC();
                    const uint32_t xp = get_write_ptr(CB_XRM);
                    const uint32_t rb = r * 32, re = spl_rows < rb + 32 ? spl_rows : rb + 32;
                    for (uint32_t row = rb; row < re; ++row) {
                        noc_async_read(x_g.get_noc_addr(dl[row], kc * XSEG), xp + (row - rb) * XSEG, XSEG);
                    }
                    noc_async_read_barrier();
                    XP_TOC(W_X_READ);
                    XP_ADD(W_X_KB, (re - rb) * XSEG / 1024);
                    push_mark(tab_, W_PUSH_NCRISC, CB_XRM, KC);
                    cb_push_back(CB_XRM, KC);
                }
            }
            // the act slices to the row owners (member pos r of the rectangle: PMK_RT_MEMBERS = g, then (x, y) per pos)
            XP_TIC();
            cb_wait_front(CB_ACTS, ACTST);
            XP_TOC(W_Y_WAIT);
            XP_TIC();
            {
                const uint32_t rp = get_read_ptr(CB_ACTS);
                const uint32_t act_base = get_write_ptr(CB_ACT);  // one block: the base, identical on every member
                for (uint32_t r = 0; r < spl_rr; ++r) {
                    const uint32_t ox = get_arg_val<uint32_t>(PMK_RT_MEMBERS + 1 + 2 * r);
                    const uint32_t oy = get_arg_val<uint32_t>(PMK_RT_MEMBERS + 2 + 2 * r);
                    for (uint32_t j = 0; j < spl_cw / 2; ++j) {
                        noc_async_write(
                            rp + (r * (spl_cw / 2) + j) * 2048,
                            get_noc_addr(ox, oy, act_base + (spl_c0 / 2 + j) * 2048),
                            2048);
                    }
                }
                noc_async_write_barrier();  // the slices landed before the owners' counts move
                for (uint32_t r = 0; r < spl_rr; ++r) {
                    const uint32_t ox = get_arg_val<uint32_t>(PMK_RT_MEMBERS + 1 + 2 * r);
                    const uint32_t oy = get_arg_val<uint32_t>(PMK_RT_MEMBERS + 2 + 2 * r);
                    noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(SEM_ACT)), 1);
                }
                noc_async_atomic_barrier();
            }
            XP_TOC(W_Y_WRITE);
            cb_pop_front(CB_ACTS, ACTST);
            if (ctl & CTL_NOROW) {
                continue;  // no row of this stream here: nothing to write
            }
        }
#else
        if (ctl & CTL_SKIP) {
            continue;  // PMK_MCAST: this core only receives the stream (no M group here)
        }
#endif
#if defined(XFFN_X_ON_NOC1) && !defined(PMK_GU_SPLIT)
        XP_TIC();
        noc_async_read(disp_g.get_noc_addr(0, d0 * 4), reinterpret_cast<uint32_t>(dl), mt * 32 * 4);
        noc_async_read_barrier();
        XP_TOC(W_DISP_READ);
#if defined(XFFN_X_SPLIT) || defined(XFFN_VALID_ROWS) || defined(XFFN_X_TILED)
        const uint32_t nr = rows;  // only the valid rows (a padding row's output is never written)
#else
        const uint32_t nr = mt * 32;  // every row of the group (padding rows read x_ext's zero row C)
#endif
        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
#ifdef XFFN_X_TILED
            // lever 2: KC tiles of every valid row from its holder's XS block, TILE -> TILE (2 pieces per tile)
            XP_TIC();
            cb_reserve_back(CB_XT, MG * KC);
            XP_TOC(W_X_WAIT);
            XP_TIC();
            {
                const uint32_t xp = get_write_ptr(CB_XT);
                for (uint32_t r = 0; r < nr; ++r) {
                    const uint32_t tok = dl[r];
                    const uint32_t h = tok >> 5, t = tok & 31u;
                    const uint32_t off = ((t >> 4) * 2) * FACE_B + (t & 15u) * PIECE;  // face (t / 16) * 2, row t % 16
                    const uint64_t src = get_noc_addr(
                        get_arg_val<uint32_t>(xs_xy + 2 * h), get_arg_val<uint32_t>(xs_xy + 2 * h + 1), xs_base);
                    const uint32_t dst = xp + ((r >> 5) * KC) * 2048;
                    const uint32_t doff = ((r & 31u) >> 4) * 2 * FACE_B + (r & 15u) * PIECE;  // dest tile row r % 32
                    for (uint32_t k = 0; k < KC; ++k) {
                        const uint64_t st = src + (kc * KC + k) * 2048 + off;
                        const uint32_t dt = dst + k * 2048 + doff;
                        noc_async_read(st, dt, PIECE);
                        noc_async_read(st + FACE_B, dt + FACE_B, PIECE);
                    }
                }
                noc_async_read_barrier();
            }
            XP_TOC(W_X_READ);
            XP_ADD(W_X_KB, nr * XSEG / 1024);
            push_mark(tab_, W_PUSH_NCRISC, CB_XT, MG * KC);
            cb_push_back(CB_XT, MG * KC);
            continue;
#endif
#ifdef XFFN_X_SPLIT
            if ((kc & 1) == 0) {
                continue;  // the reader gathers the even K chunks (NoC0) into CB_XRM
            }
            constexpr uint32_t CBX = CB_XRM1;
#else
            constexpr uint32_t CBX = CB_XRM;
#endif
            XP_TIC();
            cb_reserve_back(CBX, MG * KC);  // whole MG blocks (pushes land exactly on the CB end)
            XP_TOC(W_X_WAIT);
            XP_TIC();
            const uint32_t xp = get_write_ptr(CBX);
#ifdef PMK_XS_ROWS
            // Stage 14 B2.3: token row t from its holder's L1 row store (holder t / 32, row t % 32 at xs_base +
            // (t % 32) ROW_BYTES): the same 512-B K-chunk segment as x_ext's, over the NoC instead of from DRAM
            for (uint32_t r = 0; r < nr; ++r) {
                const uint32_t tok = TM_TOK(dl[r]);
                const uint32_t h = tok >> 5;
                const uint64_t src = get_noc_addr(
                    get_arg_val<uint32_t>(xs_xy + 2 * h),
                    get_arg_val<uint32_t>(xs_xy + 2 * h + 1),
                    xs_base + (tok & 31u) * ROW_BYTES + kc * XSEG);  // fmt: skip
                noc_async_read(src, xp + r * XSEG, XSEG);
            }
#else
            for (uint32_t r = 0; r < nr; ++r) {
                noc_async_read(x_g.get_noc_addr(TM_TOK(dl[r]), kc * XSEG), xp + r * XSEG, XSEG);
            }
#endif
            noc_async_read_barrier();
            XP_TOC(W_X_READ);
            XP_ADD(W_X_KB, nr * XSEG / 1024);
            push_mark(tab_, W_PUSH_NCRISC, CBX, MG * KC);
            cb_push_back(CBX, MG * KC);
        }
#endif
#ifdef XFFN_NO_DOWN
        continue;
#endif
#if defined(PMK_OUT_BFP8) && defined(PMK_YW_DB) && defined(PMK_YW_BRISC)
        // perf5 round 4 (PERF.md sec 16): the BRISC re-lays the groups (ffn_reader.hpp relay_poll) into the two block
        // scratches and hands each over by RELAYED (CB_TAB + 96) with a descriptor (rows, d0, nc, mt) past the
        // scratches; this RISC only issues the row writes and reports WRITTEN (+100) once they left L1 (the scratch is
        // reusable)
        {
            constexpr uint32_t RB = rowb_chunk_bytes(NC);
            constexpr uint32_t ROWB_BYTES = (H_T / NC) * RB;
            constexpr uint32_t CW = PMK_YW_COAL;
            constexpr uint32_t GRP_B = MG * 32 * CW * RB;
            const InterleavedAddrGen<true> outb_g{.bank_base_address = out_addr, .page_size = ROWB_BYTES};
            const uint32_t scr0 = (get_write_ptr(CB_WTAB) + MG * 32 * 4 + 63) & ~63u;
            volatile tt_l1_ptr uint32_t* desc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr0 + 2 * GRP_B);
            volatile tt_l1_ptr uint32_t* relayed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_ + 96);
            volatile tt_l1_ptr uint32_t* written = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_ + 100);
            for (uint32_t g_ = 0; g_ < (H_T / NC) / CW; ++g_) {
                const uint32_t k =
                    ++yw_blk;  // groups of this core in order (the reader's counter runs the same sequence)
                XP_TIC();
                while (*relayed < k) {
                    invalidate_l1_cache();
                }
                XP_TOC(W_Y_WAIT);
                XP_TIC();
                const uint32_t slot = (k - 1) & 1;
                const uint32_t rows_ = desc[slot * 4 + 0], d0_ = desc[slot * 4 + 1], nc_ = desc[slot * 4 + 2];
                const uint32_t blk = scr0 + slot * GRP_B;
                for (uint32_t row = 0; row < rows_; ++row) {
                    noc_async_write(blk + row * CW * RB, outb_g.get_noc_addr(d0_ + row, nc_ * RB), CW * RB);
                }
                noc_async_writes_flushed();  // out of the scratch before the reader may reuse it
                *written = k;
                XP_TOC(W_Y_WRITE);
                XP_ADD(W_Y_KB, rows_ * CW * SEG / 1024);
            }
        }
#elif defined(PMK_OUT_BFP8) && defined(PMK_YW_DB)
        // PERF.md sec 9-10 (the down-proj credit return): the compute pushes CB_YRM per (N chunk, tile row), nc outer;
        // the writer takes YW_COAL consecutive N chunks of all mt rows at once, re-lays every valid row's YW_COAL
        // pieces CONTIGUOUSLY (YW_COAL x RB bytes: one DRAM write per row per group instead of YW_COAL x 192-B
        // writes -- the LEARNED fan-in rule) into one of TWO group scratches (WTAB + 2 x MG 32 x YW_COAL RB), pops
        // CB_YRM right after the re-lay and defers the write barrier (a scratch is reused after its writes left L1;
        // one barrier at the phase end). Same bytes to the same addresses: numerics unchanged.
        {
            constexpr uint32_t RB = rowb_chunk_bytes(NC);
            constexpr uint32_t ROWB_BYTES = (H_T / NC) * RB;
            constexpr uint32_t CW = PMK_YW_COAL;
            static_assert((H_T / NC) % CW == 0, "coalesced N chunks divide the chunk count");
            constexpr uint32_t GRP_B = MG * 32 * CW * RB;  // one group scratch
            const InterleavedAddrGen<true> outb_g{.bank_base_address = out_addr, .page_size = ROWB_BYTES};
            const uint32_t scr0 = (get_write_ptr(CB_WTAB) + MG * 32 * 4 + 63) & ~63u;
            for (uint32_t nc = 0; nc < H_T / NC; nc += CW) {
                XP_TIC();
                cb_wait_front(CB_YRM, CW * mt * NC);
                XP_TOC(W_Y_WAIT);
                XP_TIC();
                const uint32_t rp = get_read_ptr(CB_YRM);
                const uint32_t blk = scr0 + (yw_blk & 1) * GRP_B;
                ++yw_blk;
                if (yw_blk > 2) {
                    noc_async_writes_flushed();  // this scratch's previous group left L1
                }
                for (uint32_t r = 0; r < mt; ++r) {
                    for (uint32_t t = 0; t < 32; ++t) {
                        const uint32_t row = r * 32 + t;
                        if (row >= rows) {
                            continue;
                        }
                        const uint32_t dst0 = blk + row * CW * RB;
                        const uint32_t f = (t >> 4) * 2, rr = t & 15u;
                        for (uint32_t j = 0; j < CW; ++j) {
                            volatile tt_l1_ptr uint8_t* ex =
                                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst0 + j * RB);
                            volatile tt_l1_ptr uint32_t* mw =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst0 + j * RB + ROWB_EXP_PAD);
                            for (uint32_t n = 0; n < NC; ++n) {
                                const uint32_t tile = rp + ((j * mt + r) * NC + n) * BFP8_TILE_B;
                                volatile tt_l1_ptr uint8_t* te = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile);
                                ex[2 * n] = te[f * 16 + rr];
                                ex[2 * n + 1] = te[(f + 1) * 16 + rr];
                                volatile tt_l1_ptr uint32_t* m0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    tile + BFP8_EXP_B + f * 256 + rr * 16);
                                volatile tt_l1_ptr uint32_t* m1 = m0 + 64;  // the next face (+256 B)
                                for (uint32_t w_ = 0; w_ < 4; ++w_) {
                                    mw[8 * n + w_] = m0[w_];
                                    mw[8 * n + 4 + w_] = m1[w_];
                                }
                            }
                        }
                        noc_async_write(dst0, outb_g.get_noc_addr(d0 + row, nc * RB), CW * RB);
                    }
                }
                XP_TOC(W_Y_WRITE);
                XP_ADD(W_Y_KB, (rows < mt * 32 ? rows : mt * 32) * CW * SEG / 1024);
                cb_pop_front(CB_YRM, CW * mt * NC);
            }
        }
#else
        for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
            for (uint32_t r = 0; r < mt; ++r) {
                XP_TIC();
                cb_wait_front(CB_YRM, NC);
                XP_TOC(W_Y_WAIT);
                XP_TIC();
                const uint32_t rp = get_read_ptr(CB_YRM);
#ifdef PMK_OUT_BFP8
                // row-bfp8 (moe_common.hpp): per valid row t, [NC exponent pairs | pad to 64 B][NC x (16 + 16 B)
                // mantissas] assembled in the WTAB scratch (2 slots: the previous row's write may be in flight)
                constexpr uint32_t RB = rowb_chunk_bytes(NC);
                constexpr uint32_t ROWB_BYTES = (H_T / NC) * RB;
                const InterleavedAddrGen<true> outb_g{.bank_base_address = out_addr, .page_size = ROWB_BYTES};
                const uint32_t scr0 = (get_write_ptr(CB_WTAB) + MG * 32 * 4 + 63) & ~63u;
                for (uint32_t t = 0; t < 32; ++t) {
                    const uint32_t row = r * 32 + t;
                    if (row >= rows) {
                        continue;
                    }
                    const uint32_t scr = scr0 + (t & 1) * RB;
                    if (t >= 2) {
                        noc_async_writes_flushed();  // the slot's previous row left L1
                    }
                    volatile tt_l1_ptr uint8_t* ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scr);
                    volatile tt_l1_ptr uint32_t* mw =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + ROWB_EXP_PAD);
                    const uint32_t f = (t >> 4) * 2, rr = t & 15u;
                    for (uint32_t n = 0; n < NC; ++n) {
                        const uint32_t tile = rp + n * BFP8_TILE_B;
                        volatile tt_l1_ptr uint8_t* te = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile);
                        ex[2 * n] = te[f * 16 + rr];
                        ex[2 * n + 1] = te[(f + 1) * 16 + rr];
                        volatile tt_l1_ptr uint32_t* m0 =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + BFP8_EXP_B + f * 256 + rr * 16);
                        volatile tt_l1_ptr uint32_t* m1 = m0 + 64;  // the next face (+256 B)
                        for (uint32_t w_ = 0; w_ < 4; ++w_) {
                            mw[8 * n + w_] = m0[w_];
                            mw[8 * n + 4 + w_] = m1[w_];
                        }
                    }
                    noc_async_write(scr, outb_g.get_noc_addr(d0 + row, nc * RB), RB);
                }
                noc_async_write_barrier();
#else
                for (uint32_t t = 0; t < 32; ++t) {
                    const uint32_t row = r * 32 + t;
                    if (row < rows) {
#ifdef PMK_OUT_TM
                        const uint32_t pr = dl[row];  // pair j C + tok
                        const uint32_t q = (nc * SEG) / TM_PART;
                        noc_async_write(
                            rp + t * SEG,
                            tm_g.get_noc_addr(q * TM_C + pr % TM_C, (pr / TM_C) * TM_PART + (nc * SEG) % TM_PART),
                            SEG);
#else
                        noc_async_write(rp + t * SEG, out_g.get_noc_addr(d0 + row, nc * SEG), SEG);
#endif
                    }
                }
                noc_async_write_barrier();
#endif
                XP_TOC(W_Y_WRITE);
                XP_ADD(W_Y_KB, (rows - r * 32 < 32 ? rows - r * 32 : 32) * SEG / 1024);
                cb_pop_front(CB_YRM, NC);
            }
        }
#endif  // PMK_OUT_BFP8 && PMK_YW_DB
    }
#ifdef PMK_YW_DB
    noc_async_write_barrier();  // every deferred block write landed before this core's FFN-done signal
#endif
#ifdef XFFN_PROBE
    // the writer's L1 scratch: CB_WTAB (the disp rows, done)
    XP_FLUSH(
        (get_write_ptr(CB_WTAB) + 63) & ~63u,
        get_arg_val<uint32_t>(RT0 + 4),
        get_arg_val<uint32_t>(RT0 + 5) + get_arg_val<uint32_t>(RT0 + 6));
#endif
}
