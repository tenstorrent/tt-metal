// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 speculative decoding (tt/spec_mtp.py SpecCtl; doc/ttnn/17-mtp-spec-decode/design.md §1, §4): the per-step
// control of ONE user's speculative step on ONE core per chip (every chip computes the same values: the heads' top-1
// rows are identical on every chip after their in-program candidate gather). All device-resident, no host.
//
// Token pages are [32, 64] uint32 ROW_MAJOR (one 256-B page per row; word 0 = the token), position pages [1, 32] int32
// (one 128-B page, row j = word j), the selector [1, 32] uint32 (word 0), the record [1, 64] uint32 (one 256-B page):
//   word 0 n (accepted drafts, 0..K) | 1 c (this step's base position) | 2 + j a_j (j = 0..K, the verify argmax rows)
//   | 16 + i d'_{i+1} (i = 0..K-1, the next step's drafts) | 31 steps (incremented by ACCEPT / BOOT)
//
// SPEC_MODE 1 = ACCEPT (after the verify head): a_j = vout row j; d_j = tin row j (j = 1..K); n = the longest prefix
//   with a_{j-1} == d_j; the bonus a_n -> tin row 0; sel = n; pos_v row j = c + n + 1 + j (j <= K, else -1); mpos_cu
//   row j = c + j (j <= K); mpos_ch[i] row 0 = c + n + 1 + i (i < K - 1); the record.
// SPEC_MODE 2 = BOOT (after a plain step; no verify): b = tin row 0, c = pos[0] - 1 (the plain head advanced pos to the
//   next input's position), n = 0: vout row 0 = b (the MTP catch-up's row-0 token) and the same writes as ACCEPT.
// SPEC_MODE 3 = GATHER (after the MTP chain): d'1 = mout row n, d'_{i+1} = cout_i row 0 -> tin rows 1..K, record words
//   16.. (n from the record).
//
// ACCEPT also moves the DSA rings (SPEC_REV 3, with tt/mk_mla.py rev 56): the verify wrote pool q0 + g's index ring
// rows
//   ([k | g] of the indexer, kernels/mla/ix_dm1.cpp MLA_DSA_ONE_SLOT) to ring tile rows 4 g .. 4 g + 3 of slot 0, q0 =
//   c / 4; when the next verify's pool (c + n + 1) / 4 is q0 + g with g > 0, its rows (all written by this verify, the
//   committed ones accepted) move to the canonical rows 0..3 that the decode, the prefill and the next verify read.
//   Rows past the committed position are stale and rewritten before any read, as in the plain decode.
//
// Compile-time args: K. Runtime args (fixed length NARGS: generic_op's program cache ignores runtime args):
//   ACCEPT / BOOT: tin, vout, pos (BOOT: the plain cur_pos; ACCEPT: pos_v itself), pos_v, mpos_cu, sel, rec,
//                  mpos_ch[0..2] (0 = absent), then (ACCEPT) the ring count R <= 16 and R slot-0 ring base addresses
//                  (bf16 TILE DRAM interleaved, 2 KB pages: 8 tiles per slot)
//   GATHER:        tin, mout, rec, cout[0..2] (0 = absent)
// CB 0: 1 KB scratch (64-B aligned slots).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TOK_PAGE = 256;  // [32, 64] uint32 row
constexpr uint32_t POS_PAGE = 128;  // [1, 32] int32
constexpr uint32_t REC_PAGE = 256;  // [1, 64] uint32
constexpr uint32_t W_N = 0, W_C = 1, W_A = 2, W_D = 16, W_STEPS = 31;
constexpr uint32_t RING_PAGE = 2048, RING_TILES = 8;  // a DSA ring slot: [32, 256] bf16 = 8 tiles (tt/mk_mla.py)

FORCE_INLINE uint64_t dram(uint32_t addr, uint32_t page_size, uint32_t page) {
    const InterleavedAddrGen<true> g = {.bank_base_address = addr, .page_size = page_size};
    return g.get_noc_addr(page);
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* u32(uint32_t l1) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}
FORCE_INLINE volatile tt_l1_ptr int32_t* i32(uint32_t l1) { return reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1); }
}  // namespace

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    static_assert(K >= 1 && K <= 7, "1 <= K <= 7 drafts");
    const uint32_t scr = (get_write_ptr(0) + 63u) & ~63u;
    // scratch slots (64-B aligned): [0, 256) token row, [256, 512) record, [512, 640) a position page, [640, 768) the
    // words read back, [768, 1024) the token rows' word 0 assembly
    const uint32_t s_tok = scr, s_rec = scr + 256, s_pos = scr + 512, s_aux = scr + 640;

#if SPEC_MODE == 1 || SPEC_MODE == 2
    const uint32_t tin = get_arg_val<uint32_t>(0);
    const uint32_t vout = get_arg_val<uint32_t>(1);
    const uint32_t pos_src = get_arg_val<uint32_t>(2);
    const uint32_t pos_v = get_arg_val<uint32_t>(3);
    const uint32_t mpos_cu = get_arg_val<uint32_t>(4);
    const uint32_t sel = get_arg_val<uint32_t>(5);
    const uint32_t rec = get_arg_val<uint32_t>(6);

    // the record's step counter survives (read-modify-write of the whole page)
    noc_async_read(dram(rec, REC_PAGE, 0), s_rec, REC_PAGE);
    noc_async_read(dram(pos_src, POS_PAGE, 0), s_pos, POS_PAGE);
    noc_async_read_barrier();
    const uint32_t steps = u32(s_rec)[W_STEPS] + 1;
    for (uint32_t w = 0; w < REC_PAGE / 4; ++w) {
        u32(s_rec)[w] = 0;
    }
    u32(s_rec)[W_STEPS] = steps;
    uint32_t a[K + 1];
    uint32_t n = 0;
    int32_t c;
#if SPEC_MODE == 1
    c = i32(s_pos)[0];  // pos_v row 0 = this step's base position
    for (uint32_t j = 0; j <= K; ++j) {
        noc_async_read(dram(vout, TOK_PAGE, j), s_tok, 64);
        noc_async_read_barrier();
        a[j] = u32(s_tok)[0];
    }
    bool run = true;
    for (uint32_t j = 1; j <= K; ++j) {
        noc_async_read(dram(tin, TOK_PAGE, j), s_tok, 64);
        noc_async_read_barrier();
        const uint32_t d = u32(s_tok)[0];
        if (run && a[j - 1] == d) {
            n = j;
        } else {
            run = false;
        }
    }
#else
    c = i32(s_pos)[0] - 1;  // the plain head advanced pos to the next input's position
    noc_async_read(dram(tin, TOK_PAGE, 0), s_tok, 64);
    noc_async_read_barrier();
    a[0] = u32(s_tok)[0];
    for (uint32_t j = 1; j <= K; ++j) {
        a[j] = a[0];
    }
    // the MTP catch-up's token rows: row 0 = b (rows 1..K are never read for n = 0: their MTP positions > c)
    u32(s_tok)[0] = a[0];
    noc_async_write(s_tok, dram(vout, TOK_PAGE, 0), 64);
#endif
    u32(s_rec)[W_N] = n;
    u32(s_rec)[W_C] = static_cast<uint32_t>(c);
    for (uint32_t j = 0; j <= K; ++j) {
        u32(s_rec)[W_A + j] = a[j];
    }
    noc_async_write(s_rec, dram(rec, REC_PAGE, 0), REC_PAGE);
    // the bonus = the next step's row-0 input (word 0 of tin row 0; the other words of the row are not read)
    u32(s_aux)[0] = a[n];
    noc_async_write(s_aux, dram(tin, TOK_PAGE, 0), 64);
    // the selector page: word 0 = n (the KDA candidate row the next verify starts from; the MTP chain's row)
    noc_async_write_barrier();
    for (uint32_t w = 0; w < 16; ++w) {
        u32(s_aux)[w] = w == 0 ? n : 0;
    }
    noc_async_write(s_aux, dram(sel, POS_PAGE, 0), 64);
    noc_async_write_barrier();
    // positions: the next verify (rows 0..K), this step's MTP catch-up (rows 0..K), the chain rows
    for (uint32_t j = 0; j < 32; ++j) {
        i32(s_pos)[j] = j <= K ? c + static_cast<int32_t>(n + 1 + j) : -1;
    }
    noc_async_write(s_pos, dram(pos_v, POS_PAGE, 0), POS_PAGE);
    noc_async_write_barrier();
    for (uint32_t j = 0; j < 32; ++j) {
        i32(s_pos)[j] = j <= K ? c + static_cast<int32_t>(j) : -1;
    }
    noc_async_write(s_pos, dram(mpos_cu, POS_PAGE, 0), POS_PAGE);
    noc_async_write_barrier();
    for (uint32_t i = 0; i + 1 < K && i < 3; ++i) {
        const uint32_t mp = get_arg_val<uint32_t>(7 + i);
        if (mp == 0) {
            continue;
        }
        for (uint32_t j = 0; j < 32; ++j) {
            i32(s_pos)[j] = j == 0 ? c + static_cast<int32_t>(n + 1 + i) : -1;
        }
        noc_async_write(s_pos, dram(mp, POS_PAGE, 0), POS_PAGE);
        noc_async_write_barrier();
    }
#if SPEC_MODE == 1
    // the DSA rings (header): rows 4 g .. 4 g + 3 -> rows 0..3 of each of the slot's 8 ring tiles (faces 0 / 1: 128 B
    // each)
    const uint32_t n_ring = get_arg_val<uint32_t>(10);
    const uint32_t grp =
        c < 0 ? 0u : static_cast<uint32_t>(c + static_cast<int32_t>(n) + 1) / 4 - static_cast<uint32_t>(c) / 4;
    if (grp > 0 && n_ring > 0) {
        const uint32_t r0 = 4 * grp;
        const uint32_t off = (r0 >> 4) * 1024 + (r0 & 15) * 32;  // kernels/mla/mla_common.hpp row_off
        for (uint32_t l = 0; l < n_ring && l < 16; ++l) {
            const uint32_t base = get_arg_val<uint32_t>(11 + l);
            for (uint32_t i = 0; i < RING_TILES; ++i) {
                const uint64_t a = dram(base, RING_PAGE, i);
                noc_async_read(a + off, s_tok, 128);
                noc_async_read(a + off + 512, s_tok + 128, 128);
                noc_async_read_barrier();
                noc_async_write(s_tok, a, 128);
                noc_async_write(s_tok + 128, a + 512, 128);
                noc_async_write_barrier();
            }
        }
    }
#endif
#elif SPEC_MODE == 3
    const uint32_t tin = get_arg_val<uint32_t>(0);
    const uint32_t mout = get_arg_val<uint32_t>(1);
    const uint32_t rec = get_arg_val<uint32_t>(2);
    noc_async_read(dram(rec, REC_PAGE, 0), s_rec, REC_PAGE);
    noc_async_read_barrier();
    const uint32_t n = u32(s_rec)[W_N];
    uint32_t d[K];
    noc_async_read(dram(mout, TOK_PAGE, n), s_tok, 64);
    noc_async_read_barrier();
    d[0] = u32(s_tok)[0];
    for (uint32_t i = 1; i < K; ++i) {
        const uint32_t co = get_arg_val<uint32_t>(3 + (i - 1));
        noc_async_read(dram(co, TOK_PAGE, 0), s_tok, 64);
        noc_async_read_barrier();
        d[i] = u32(s_tok)[0];
    }
    for (uint32_t i = 0; i < K; ++i) {
        u32(s_rec)[W_D + i] = d[i];
        u32(s_aux + 64 * (i % 4))[0] = d[i];
        noc_async_write(s_aux + 64 * (i % 4), dram(tin, TOK_PAGE, 1 + i), 64);
        if (i % 4 == 3) {
            noc_async_write_barrier();
        }
    }
    noc_async_write(s_rec, dram(rec, REC_PAGE, 0), REC_PAGE);
    noc_async_write_barrier();
#else
#error "SPEC_MODE: 1 ACCEPT, 2 BOOT, 3 GATHER"
#endif
}
