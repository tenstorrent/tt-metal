// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 DFlash2 in-trace drafter (tt/dflash2_kind.py): the two per-step control kernels on ONE core per chip
// (every chip computes the same values: the record and the token page are replicated). All device-resident, no host.
//
// Pages: token page tin [32, 64] uint32 ROW_MAJOR (256-B rows, word 0 = the token); record rec [1, 64] uint32 (256 B:
// word 0 n, word 1 c, words 16 + i = the next step's drafts: tt/spec_mtp.py SpecCtl's layout); the drafter's pages
// ids [1, 32] uint32 (128 B), bonus16 [1, 16] uint32 (64 B), bpos / cpos / cur [32] int32 (128 B each), bposu / cposu
// [1, 32] uint32 (128 B each).
//
// DF2_MODE 1 = PREP (after SpecCtl ACCEPT / BOOT): n = rec[0], c = rec[1], bonus = tin row 0 word 0; c' = c + n + 1.
//   ids = [bonus, MASK_ID x K, 0 ...]; bonus16 = bonus x 16; cpos / cposu = c + j (this step's verify rows: the
//   context K / V write positions); bpos / bposu = c' + j (the block rows); cur = c' + K (the block's attention bound,
//   every row). Positions are clamped at 0 (SpecMTP.prepare runs the step at far-negative positions: the gathers
//   stay in range and the cache writes land on position 0, which the real context writes overwrite).
// DF2_MODE 2 = SELECT (after the selector scores): scores [32 (block row), 32, 32] fp32 TILE (one 4-KB tile per block
//   row: [prev candidate p, candidate c], 16 x 16 valid), cand [32, 16] uint32 ROW_MAJOR (64-B rows: the global vocab
//   ids of the row's top-16). The greedy walk from the bonus: prev = 0; for s = 1..K: j = argmax_c scores[s][prev][c]
//   (first maximum), tok = cand[s][j], prev = j -> tin row s word 0 = tok, rec word 16 + s - 1 = tok.
//
// Compile-time args: K, MASK_ID, HD (the rung-3 cos / sin tiles' head dim; 0 = none). Runtime args (fixed length
// 24; generic_op's program cache ignores runtime args):
//   PREP:   tin, rec, ids, bonus16, bpos, bposu, cpos, cposu, cur [, cos_tab, sin_tab, cosb, sinb, cosc, sinc, cap,
//           cur_g, G]: cur_g = the bound again as a [1, G] int32 page (the SDPA's per-kv-group batch rows, rung 3)
//           rung 3 (tt/dflash2_mk.py, args 9..15; cos_tab == 0: off): the ROW_MAJOR [cap, HD] bf16 tables -> the
//           [1, 1, 32, HD] bf16 TILE cos / sin tiles of the block rows' positions (cosb / sinb) and of this step's
//           rows' (cosc / sinc), rows clamped to [0, cap), re-laid into the tile faces by the RISC
//           the sequence-sharded cache (define DF2_SHARD = C, tt/dflash2_1m.py): 17 rank page, 18 cpos_l, 19 bpos_l,
//           20 cur_l (the rows' local positions / the local bound), 21 local_cap (the scratch rows' base)
//   SELECT: scores, cand, tin, rec
// CB 0: 20.5 KB + 64 scratch.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TOK_PAGE = 256, REC_PAGE = 256, POS_PAGE = 128, B16_PAGE = 64, CAND_PAGE = 64, TILE_PAGE = 4096;
constexpr uint32_t W_N = 0, W_C = 1, W_D = 16;

FORCE_INLINE uint64_t dram(uint32_t addr, uint32_t page_size, uint32_t page) {
    const InterleavedAddrGen<true> g = {.bank_base_address = addr, .page_size = page_size};
    return g.get_noc_addr(page);
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* u32(uint32_t l1) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}
FORCE_INLINE volatile tt_l1_ptr int32_t* i32(uint32_t l1) { return reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1); }

// element (r, c) of a 32 x 32 fp32 tile (faces 16 x 16: f0 rows 0-15 / cols 0-15, f1 rows 0-15 / cols 16-31, ...)
FORCE_INLINE uint32_t tile_idx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
// byte offset of row r's left face row (16 bf16) in a bf16 tile (the right face row is + 512)
FORCE_INLINE uint32_t row_off(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }
FORCE_INLINE void copy32(uint32_t src, uint32_t dst) {
    volatile tt_l1_ptr uint32_t* s = u32(src);
    volatile tt_l1_ptr uint32_t* d = u32(dst);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t MASK_ID = get_compile_time_arg_val(1);
    constexpr uint32_t HD = get_compile_time_arg_val(2);
    static_assert(K >= 1 && K <= 7, "1 <= K <= 7 drafts");
    const uint32_t scr = (get_write_ptr(0) + 63u) & ~63u;
    // scratch slots (64-B aligned): [0, 256) a token row / the record, [256, 512) a page, [512, 4608) a tile
    const uint32_t s_rec = scr, s_pg = scr + 256, s_tile = scr + 512;

#if DF2_MODE == 1
    const uint32_t tin = get_arg_val<uint32_t>(0);
    const uint32_t rec = get_arg_val<uint32_t>(1);
    const uint32_t ids = get_arg_val<uint32_t>(2);
    const uint32_t bonus16 = get_arg_val<uint32_t>(3);
    const uint32_t bpos = get_arg_val<uint32_t>(4);
    const uint32_t bposu = get_arg_val<uint32_t>(5);
    const uint32_t cpos = get_arg_val<uint32_t>(6);
    const uint32_t cposu = get_arg_val<uint32_t>(7);
    const uint32_t cur = get_arg_val<uint32_t>(8);

    noc_async_read(dram(rec, REC_PAGE, 0), s_rec, REC_PAGE);
    noc_async_read(dram(tin, TOK_PAGE, 0), s_pg, 64);
    noc_async_read_barrier();
    const int32_t n = i32(s_rec)[W_N];
    const int32_t c = i32(s_rec)[W_C];
    const uint32_t bonus = u32(s_pg)[0];
    const int32_t c1 = c + n + 1;
    // ids [bonus, mask x K, 0 ...]
    for (uint32_t j = 0; j < 32; ++j) {
        u32(s_pg)[j] = j == 0 ? bonus : (j <= K ? MASK_ID : 0u);
    }
    noc_async_write(s_pg, dram(ids, POS_PAGE, 0), POS_PAGE);
    noc_async_write_barrier();
    for (uint32_t j = 0; j < 16; ++j) {
        u32(s_pg)[j] = bonus;
    }
    noc_async_write(s_pg, dram(bonus16, B16_PAGE, 0), B16_PAGE);
    noc_async_write_barrier();
    // this step's rows: c + j (the context write); as uint32 for the RoPE gather
    for (uint32_t j = 0; j < 32; ++j) {
        const int32_t p = c + static_cast<int32_t>(j);
        i32(s_pg)[j] = p < 0 ? 0 : p;
    }
    noc_async_write(s_pg, dram(cpos, POS_PAGE, 0), POS_PAGE);
    noc_async_write(s_pg, dram(cposu, POS_PAGE, 0), POS_PAGE);
    noc_async_write_barrier();
    // the block rows: c' + j
    for (uint32_t j = 0; j < 32; ++j) {
        const int32_t p = c1 + static_cast<int32_t>(j);
        i32(s_pg)[j] = p < 0 ? 0 : p;
    }
    noc_async_write(s_pg, dram(bpos, POS_PAGE, 0), POS_PAGE);
    noc_async_write(s_pg, dram(bposu, POS_PAGE, 0), POS_PAGE);
    noc_async_write_barrier();
    // the block's attention bound (every row attends positions <= c' + K: the context and the whole block)
    {
        const int32_t b = c1 + static_cast<int32_t>(K);
        for (uint32_t j = 0; j < 32; ++j) {
            i32(s_pg)[j] = b < 0 ? 0 : b;
        }
    }
    noc_async_write(s_pg, dram(cur, POS_PAGE, 0), POS_PAGE);
    noc_async_write_barrier();
#if defined(DF2_SHARD) && DF2_SHARD > 0
    // the sequence-sharded drafter cache (tt/dflash2_1m.py DrafterShardMap, GLM53_DFLASH2_SHARD=1): C = DF2_SHARD
    // chips, unit 128 tokens, unit u on chip u % C, local position (u / C) 128 + p % 128. This chip's rank comes from
    // its rank page (a sharded page: word 0 = the rank). cpos_l / bpos_l = this step's rows' / the block rows' LOCAL
    // positions (a row another chip owns -> the scratch rows local_cap + j, written, never read); cur_l = the local
    // inclusive bound of the block attention (-1 when this chip owns no row <= c' + K: the SDPA skips such rows).
    // Args: 17 rank page, 18 cpos_l, 19 bpos_l, 20 cur_l, 21 local_cap.
    {
        const uint32_t rank_pg = get_arg_val<uint32_t>(17), cpos_l = get_arg_val<uint32_t>(18);
        const uint32_t bpos_l = get_arg_val<uint32_t>(19), cur_l = get_arg_val<uint32_t>(20);
        const int32_t local_cap = static_cast<int32_t>(get_arg_val<uint32_t>(21));
        constexpr int32_t C = DF2_SHARD, UNIT = 128;
        noc_async_read(dram(rank_pg, POS_PAGE, 0), s_pg, POS_PAGE);
        noc_async_read_barrier();
        const int32_t r = i32(s_pg)[0];
        for (uint32_t set = 0; set < 2; ++set) {
            const int32_t base = set == 0 ? c : c1;
            for (uint32_t j = 0; j < 32; ++j) {
                int32_t p = base + static_cast<int32_t>(j);
                p = p < 0 ? 0 : p;
                const int32_t u = p / UNIT;
                i32(s_pg)[j] = (u % C == r) ? (u / C) * UNIT + p % UNIT : local_cap + static_cast<int32_t>(j);
            }
            noc_async_write(s_pg, dram(set == 0 ? cpos_l : bpos_l, POS_PAGE, 0), POS_PAGE);
            noc_async_write_barrier();
        }
        {
            int32_t n = c1 + static_cast<int32_t>(K) + 1;  // rows 0 .. cur (cur = c' + K)
            n = n < 0 ? 0 : n;
            const int32_t fu = n / UNIT, rem = n % UNIT;
            const int32_t full = fu > r ? (fu - r + C - 1) / C : 0;
            const int32_t nloc = full * UNIT + ((fu % C == r) ? rem : 0);
            for (uint32_t j = 0; j < 32; ++j) {
                i32(s_pg)[j] = nloc - 1;
            }
            noc_async_write(s_pg, dram(cur_l, POS_PAGE, 0), POS_PAGE);
            noc_async_write_barrier();
        }
    }
#endif
    // rung 3: the bound as a [1, G] page (G <= 4 entries: one 16-B write into the page)
    if (get_arg_val<uint32_t>(16) != 0) {
        const uint32_t gbytes = (get_arg_val<uint32_t>(17) * 4 + 15u) & ~15u;  // G <= 32 entries
        noc_async_write(s_pg, dram(get_arg_val<uint32_t>(16), gbytes, 0), gbytes);
        noc_async_write_barrier();
    }
    // rung 3: the cos / sin tiles of the block rows (c' + j) and of this step's rows (c + j)
    const uint32_t cos_tab = get_arg_val<uint32_t>(9), sin_tab = get_arg_val<uint32_t>(10);
    if constexpr (HD > 0) {
        if (cos_tab != 0) {
            constexpr uint32_t ROW_BYTES = 2 * HD, HT = HD / 32;
            const uint32_t cap = get_arg_val<uint32_t>(15);
            const uint32_t s_rows = scr + 4608, s_tiles = s_rows + 32 * ROW_BYTES;
            for (uint32_t set = 0; set < 2; ++set) {
                const int32_t base = set == 0 ? c1 : c;
                for (uint32_t which = 0; which < 2; ++which) {
                    const uint32_t tab = which == 0 ? cos_tab : sin_tab;
                    const uint32_t out = get_arg_val<uint32_t>(11 + 2 * set + which);
                    for (uint32_t t = 0; t < 32; ++t) {
                        int32_t p = base + static_cast<int32_t>(t);
                        p = p < 0 ? 0 : p;
                        const uint32_t pu = static_cast<uint32_t>(p) >= cap ? cap - 1 : static_cast<uint32_t>(p);
                        noc_async_read(dram(tab, ROW_BYTES, pu), s_rows + t * ROW_BYTES, ROW_BYTES);
                    }
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                    for (uint32_t t = 0; t < 32; ++t) {
                        for (uint32_t k = 0; k < HT; ++k) {
                            for (uint32_t f = 0; f < 2; ++f) {
                                copy32(
                                    s_rows + t * ROW_BYTES + k * 64 + f * 32,
                                    s_tiles + k * TILE_PAGE / 2 + row_off(t) + f * 512);
                            }
                        }
                    }
                    invalidate_l1_cache();
                    for (uint32_t k = 0; k < HT; ++k) {
                        noc_async_write(s_tiles + k * TILE_PAGE / 2, dram(out, TILE_PAGE / 2, k), TILE_PAGE / 2);
                    }
                    noc_async_write_barrier();
                }
            }
        }
    }
#else
    const uint32_t scores = get_arg_val<uint32_t>(0);
    const uint32_t cand = get_arg_val<uint32_t>(1);
    const uint32_t tin = get_arg_val<uint32_t>(2);
    const uint32_t rec = get_arg_val<uint32_t>(3);

    noc_async_read(dram(rec, REC_PAGE, 0), s_rec, REC_PAGE);
    noc_async_read_barrier();
    uint32_t prev = 0;
    for (uint32_t s = 1; s <= K; ++s) {
        noc_async_read(dram(scores, TILE_PAGE, s), s_tile, TILE_PAGE);
        noc_async_read(dram(cand, CAND_PAGE, s), s_pg, CAND_PAGE);
        noc_async_read_barrier();
        volatile tt_l1_ptr float* t = reinterpret_cast<volatile tt_l1_ptr float*>(s_tile);
        uint32_t best = 0;
        float bv = t[tile_idx(prev, 0)];
        for (uint32_t cc = 1; cc < 16; ++cc) {
            const float v = t[tile_idx(prev, cc)];
            if (v > bv) {
                bv = v;
                best = cc;
            }
        }
        const uint32_t tok = u32(s_pg)[best];
        u32(s_rec)[W_D + s - 1] = tok;
        // the token row (word 0; the row's other words are never read)
        for (uint32_t w = 0; w < 16; ++w) {
            u32(s_pg)[w] = w == 0 ? tok : 0u;
        }
        noc_async_write(s_pg, dram(tin, TOK_PAGE, s), 64);
        noc_async_write_barrier();
        prev = best;
    }
    noc_async_write(s_rec, dram(rec, REC_PAGE, 0), REC_PAGE);
    noc_async_write_barrier();
#endif
}
