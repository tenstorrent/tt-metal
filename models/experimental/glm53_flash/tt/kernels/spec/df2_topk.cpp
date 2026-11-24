// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 DFlash2 (tt/dflash2_kind.py TtnnOps.topk_rows, GLM53_DF2_TOPK=kernel): the exact fp32 top-K of the block's
// draft rows over this chip's vocab slice of the lm_head logits, as TWO one-kernel launches instead of ttnn.topk over
// the whole [32, V] tensor (the 2026-10-23 02:5xZ silicon breakdown: the 22-launch head chain is 13.1 of the block's
// 17.5 ms at TP2, V = 77,440 columns = 2,420 tiles, 32 rows of which only rows 1 .. KR (the K mask rows) draft).
//   TOPK_MODE 1 (SCAN, NC cores): core i takes column tiles [t0, t0 + n) of the fp32 TILE logits [1, 1, 32, V] (page =
//     column tile, 4 KB); for every row r in 1 .. KR it keeps the TOPK best (value, column) of its tiles -- values
//     compared as order-preserving int32 keys (sign-magnitude -> two's complement: key = bits >= 0 ? bits : bits ^
//     0x7FFFFFFF), ties to the lower column -- and writes them as a 1-KB page [KR x TOPK x (bits, column)] of the
//     partials tensor ([NC, 256] uint32 ROW_MAJOR; row 0 of the block is unused).
//   TOPK_MODE 2 (MERGE, one core): the NC pages -> per row the TOPK best of the NC x TOPK candidates (the same keys,
//     ties to the lower column) -> vals (fp32 TILE [1, 1, 32, 32]: rows 1 .. KR, columns 0 .. TOPK-1, descending; the
//     other elements -FLT_MAX) and ids (uint32 TILE [1, 1, 32, 32]: this chip's local vocab ids; 0 elsewhere):
//     the tensors the kind's global top-K / selector chain takes from ttnn.topk.
// Compile-time args: KR, TOPK, NC, TILES (column tiles of the logits), BATCH (tiles read per DRAM round trip).
// Runtime args: SCAN: logits, partials, t0, n; MERGE: partials, vals, ids.
// CB 0: scratch (SCAN: BATCH tiles + the lists; MERGE: NC KB + 2 output tiles).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t KR = get_compile_time_arg_val(0);
constexpr uint32_t TOPK = get_compile_time_arg_val(1);
constexpr uint32_t NC = get_compile_time_arg_val(2);
constexpr uint32_t TILES = get_compile_time_arg_val(3);
constexpr uint32_t BATCH = get_compile_time_arg_val(4);
constexpr uint32_t TB32 = 4096;
constexpr uint32_t PAGE = KR * TOPK * 2 * 4;  // the partial page: KR x TOPK x (bits, column) words
static_assert(KR >= 1 && KR <= 31 && TOPK >= 1 && TOPK <= 32 && PAGE <= 4096, "KR x TOPK x 8 B <= 4 KB");

FORCE_INLINE uint64_t dram(uint32_t addr, uint32_t page_size, uint32_t page) {
    const InterleavedAddrGen<true> g = {.bank_base_address = addr, .page_size = page_size};
    return g.get_noc_addr(page);
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* u32(uint32_t l1) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}
// element offset of tile element (r, c) in the face-major layout of a 32x32 tile
FORCE_INLINE uint32_t tile_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
// order-preserving int32 key of an fp32 bit pattern (NaN sorts with its sign; -0 < +0)
FORCE_INLINE int32_t key_of(uint32_t bits) {
    return static_cast<int32_t>((bits & 0x80000000u) ? (bits ^ 0x7FFFFFFFu) : bits);
}

// a descending list of TOPK (key, column) at l1 (2 words per entry); insert (k, col) if it beats the last entry;
// equal keys keep the earlier (lower column, since columns arrive in increasing order per row and the merge takes
// cores in order) entry ahead
FORCE_INLINE void insert(uint32_t l1, int32_t k, uint32_t col) {
    volatile tt_l1_ptr int32_t* e = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    if (k <= e[2 * (TOPK - 1)]) {
        return;
    }
    uint32_t p = TOPK - 1;
    while (p > 0 && k > e[2 * (p - 1)]) {
        e[2 * p] = e[2 * (p - 1)];
        e[2 * p + 1] = e[2 * (p - 1) + 1];
        --p;
    }
    e[2 * p] = k;
    e[2 * p + 1] = static_cast<int32_t>(col);
}

FORCE_INLINE void clear_lists(uint32_t l1) {
    volatile tt_l1_ptr int32_t* e = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    for (uint32_t i = 0; i < KR * TOPK; ++i) {
        e[2 * i] = static_cast<int32_t>(0x80000000u);  // the smallest key (nothing beats it until written)
        e[2 * i + 1] = 0;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t scr = (get_write_ptr(0) + 63u) & ~63u;
#if TOPK_MODE == 1
    const uint32_t logits = get_arg_val<uint32_t>(0);
    const uint32_t partials = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    const uint32_t s_tiles = scr;                // BATCH fp32 tiles
    const uint32_t s_list = scr + BATCH * TB32;  // KR lists of TOPK (key, column)
    clear_lists(s_list);
    for (uint32_t b = 0; b < n; b += BATCH) {
        const uint32_t m = n - b < BATCH ? n - b : BATCH;
        for (uint32_t j = 0; j < m; ++j) {
            noc_async_read(dram(logits, TB32, t0 + b + j), s_tiles + j * TB32, TB32);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t j = 0; j < m; ++j) {
            volatile tt_l1_ptr uint32_t* t = u32(s_tiles + j * TB32);
            const uint32_t col0 = (t0 + b + j) * 32;
            for (uint32_t r = 1; r <= KR; ++r) {
                const uint32_t list = s_list + (r - 1) * TOPK * 8;
                for (uint32_t c = 0; c < 32; ++c) {
                    insert(list, key_of(t[tile_off(r, c)]), col0 + c);
                }
            }
        }
    }
    // the keys back to fp32 bits for the merge's page (the merge re-keys); page = core
    {
        volatile tt_l1_ptr uint32_t* e = u32(s_list);
        for (uint32_t i = 0; i < KR * TOPK; ++i) {
            const uint32_t k = e[2 * i];
            e[2 * i] = (k & 0x80000000u) ? (k ^ 0x7FFFFFFFu) : k;
        }
    }
    invalidate_l1_cache();
    noc_async_write(s_list, dram(partials, PAGE, core), PAGE);
    noc_async_write_barrier();
#else
    const uint32_t partials = get_arg_val<uint32_t>(0);
    const uint32_t vals = get_arg_val<uint32_t>(1);
    const uint32_t ids = get_arg_val<uint32_t>(2);
    const uint32_t s_parts = scr;                 // NC pages
    const uint32_t s_list = s_parts + NC * PAGE;  // KR lists
    const uint32_t s_vals = (s_list + KR * TOPK * 8 + 63u) & ~63u;
    const uint32_t s_ids = s_vals + TB32;
    for (uint32_t i = 0; i < NC; ++i) {
        noc_async_read(dram(partials, PAGE, i), s_parts + i * PAGE, PAGE);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    clear_lists(s_list);
    for (uint32_t r = 1; r <= KR; ++r) {
        const uint32_t list = s_list + (r - 1) * TOPK * 8;
        for (uint32_t i = 0; i < NC; ++i) {  // cores in order: equal keys keep the lower column ahead
            volatile tt_l1_ptr uint32_t* e = u32(s_parts + i * PAGE + (r - 1) * TOPK * 8);
            for (uint32_t q = 0; q < TOPK; ++q) {
                insert(list, key_of(e[2 * q]), e[2 * q + 1]);
            }
        }
    }
    // the output tiles: zero, then rows 1 .. KR columns 0 .. TOPK-1
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < 2 * TB32; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, s_vals + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* v = u32(s_vals);
        volatile tt_l1_ptr uint32_t* d = u32(s_ids);
        for (uint32_t i = 0; i < 1024; ++i) {
            v[i] = 0xFF7FFFFFu;  // -FLT_MAX: the padding never enters the global top-K
        }
        for (uint32_t r = 1; r <= KR; ++r) {
            volatile tt_l1_ptr uint32_t* e = u32(s_list + (r - 1) * TOPK * 8);
            for (uint32_t q = 0; q < TOPK; ++q) {
                const uint32_t k = e[2 * q];
                v[tile_off(r, q)] = (k & 0x80000000u) ? (k ^ 0x7FFFFFFFu) : k;  // the fp32 bits back
                d[tile_off(r, q)] = e[2 * q + 1];
            }
        }
    }
    invalidate_l1_cache();
    noc_async_write(s_vals, dram(vals, TB32, 0), TB32);
    noc_async_write(s_ids, dram(ids, TB32, 0), TB32);
    noc_async_write_barrier();
#endif
}
