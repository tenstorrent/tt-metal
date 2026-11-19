// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK cache-write phase, DSA indexer rows (sparse_dsa layers; tt/pmk_cache.py dsa_parts):
//   (a) pooled keys: pooled [T / 4, 128] bf16 TILE (one pooled key per 4 tokens: tile row j = tokens [128 j, 128 j +
//       128) = one pooled-key store page of POOL_TILES tiles) -> the slot's store pages [B x PPS, 1, 32, 128]
//       (page = slot PPS + pos / 128: prefill_tp.py DsaIndexer.store / write_packed pool_idx). Whole pages; a page
//       with no valid row is a pad page (skipped). This core's pages: [first, first + count).
//   (b) the decode indexer's ring (ring [B, 1, 32, 256] bf16 TILE: slot s = tiles [8 s, 8 s + 8)): after a segment
//       of positions [pos0, end), ring row j (< 4) = the [k | g] row of position 4 (end / 4) + j if < end, else of
//       4 (end / 4) - 4 + j (write_packed's rr: a position outside the segment -> a zero row); rows 4..31 zero. kg
//       [T, 256] bf16 TILE = every packed row's [k | g] (rows = tokens). The core with ring_role = 1 writes every
//       segment's ring (8 tiles: the tile row built in scratch from the kg face rows, then written whole).
// SRC_MODE 0: pooled / kg are tensors. SRC_MODE 1 (§4b H2 layout): kg block j (32 rows = 8 tiles) at kg_addr and the
// pooled-key page j' (tokens [128 j', +128) = 4 tiles) at pooled_addr, both on the MM core that holds row block 4 j'
// (resp. j); the (x, y) pairs are runtime args 10 + 2 j (j from src_row0 / 32); the accessors are unused.
// Compile-time (PH_CT0 +): CB_SCR, SEM_WAIT, WAIT_MODE, WAIT_COUNT, NSEG_MAX, POOL_TILES (4), KG_TILES (8), PPS
//   (store pages per slot), RING_TILES (8), SRC_MODE, then the TensorAccessorArgs of pooled, store, kg, ring, segtab.
// Runtime (ph_rt0 +): pooled (SRC_MODE 1: pooled_addr), store, kg (kg_addr), ring, segtab addresses, src_row0,
//   src_rows, first page, page count, ring_role, then (SRC_MODE 1) src_rows / 32 (x, y) pairs.
// Scratch CB_SCR: max(POOL_TILES, 2 x RING_TILES) x 2048 + align64(table).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "cache_common.hpp"

namespace {
FORCE_INLINE void copy32(uint32_t src, uint32_t dst) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t SEM_WAIT = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t WAIT_MODE = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t NSEG_MAX = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t POOL_TILES = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t KG_TILES = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t PPS = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t RING_TILES = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t SRC_MODE = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr auto a_args = TensorAccessorArgs<PH_CT0 + 10>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    const auto pooled = TensorAccessor(a_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto store = TensorAccessor(b_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto kg = TensorAccessor(c_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto ring = TensorAccessor(d_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const auto tab = TensorAccessor(t_args, get_arg_val<uint32_t>(ph_rt0 + 4));
    const uint32_t src_row0 = get_arg_val<uint32_t>(ph_rt0 + 5);
    const uint32_t src_rows = get_arg_val<uint32_t>(ph_rt0 + 6);
    const uint32_t first = get_arg_val<uint32_t>(ph_rt0 + 7);
    const uint32_t count = get_arg_val<uint32_t>(ph_rt0 + 8);
    const uint32_t ring_role = get_arg_val<uint32_t>(ph_rt0 + 9);
    constexpr uint32_t TB = pmk::TILE_BF16;
    constexpr uint32_t POOL_ROWS = 128;  // tokens per pooled-key page (4 x 32 pools)
    constexpr uint32_t KPOOL = 4;
    constexpr uint32_t NSLOT = POOL_TILES > 2 * RING_TILES ? POOL_TILES : 2 * RING_TILES;
    static_assert(KG_TILES == RING_TILES, "a ring tile row is as wide as a [k | g] row");
    if (count == 0 && ring_role == 0) {
        return;
    }

    const uint32_t scr = get_write_ptr(CB_SCR);
    const uint32_t tab_l1 = scr + NSLOT * TB;
    if constexpr (WAIT_MODE == pmk::WAIT_ALL) {
        pmk::wait_min(SEM_WAIT, WAIT_COUNT);
    }
    const pmk::SegTable st{pmk::load_segtab(tab, tab_l1)};

    // (a) pooled-key pages
    for (uint32_t j = first; j < first + count; ++j) {
        const uint32_t r = j * POOL_ROWS;
        if (r < src_row0 || r - src_row0 >= src_rows) {
            continue;
        }
        const uint32_t k = st.seg_of_row(r);
        if (k == pmk::SEG_NONE || !st.valid(k, r)) {
            continue;  // a pad page
        }
        if constexpr (WAIT_MODE == pmk::WAIT_BLOCKS) {
            pmk::wait_min(SEM_WAIT, (r - src_row0) / 32 + POOL_ROWS / 32);  // the page's 4 token blocks pooled
        }
        const uint32_t dst = st.seg(k, pmk::S_SLOT) * PPS + st.pos(k, r) / POOL_ROWS;
        const uint32_t s0 = ((r - src_row0) / POOL_ROWS) * POOL_TILES, d0 = dst * POOL_TILES;
        for (uint32_t c = 0; c < POOL_TILES; ++c) {
            if constexpr (SRC_MODE == 1) {
                const uint32_t blk = ((r - src_row0) / 32);
                const uint32_t x = get_arg_val<uint32_t>(ph_rt0 + 10 + 2 * blk),
                               y = get_arg_val<uint32_t>(ph_rt0 + 11 + 2 * blk);
                noc_async_read(get_noc_addr(x, y, get_arg_val<uint32_t>(ph_rt0 + 0) + c * TB), scr + c * TB, TB);
            } else {
                noc_async_read_page(s0 + c, pooled, scr + c * TB);
            }
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < POOL_TILES; ++c) {
            noc_async_write_page(d0 + c, store, scr + c * TB);
        }
        noc_async_write_barrier();
    }
    if (ring_role == 0) {
        return;
    }

    // (b) rings: scratch [0, RING_TILES) = the ring tile row being built, [RING_TILES, 2 RING_TILES) = a kg row-tile
    const uint32_t kg_l1 = scr + RING_TILES * TB;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    if constexpr (WAIT_MODE == pmk::WAIT_BLOCKS) {
        pmk::wait_min(SEM_WAIT, src_rows / 32);  // every local [k | g] row has landed
    }
    const uint32_t n = st.n();
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t slot = st.seg(k, pmk::S_SLOT);
        if (slot == pmk::SEG_NONE) {
            continue;
        }
        const uint32_t pos0 = st.seg(k, pmk::S_POS0), nv = st.seg(k, pmk::S_NVALID), row0 = st.seg(k, pmk::S_ROW0);
        const uint32_t end = pos0 + nv, base = KPOOL * (end / KPOOL);
        for (uint32_t off = 0; off < RING_TILES * TB; off += MEM_ZEROS_SIZE) {
            const uint32_t nb = MEM_ZEROS_SIZE < RING_TILES * TB - off ? MEM_ZEROS_SIZE : RING_TILES * TB - off;
            noc_async_read(zeros, scr + off, nb);
        }
        noc_async_read_barrier();
        uint32_t loaded = pmk::SEG_NONE;  // the kg row-tile in scratch
        for (uint32_t j = 0; j < KPOOL; ++j) {
            uint32_t pos;
            if (base + j < end) {
                pos = base + j;
            } else if (base >= KPOOL) {
                pos = base - KPOOL + j;
            } else {
                continue;
            }
            if (pos < pos0 || pos - pos0 >= nv) {
                continue;  // outside the segment: a zero row (write_packed's rr default)
            }
            const uint32_t row = row0 + (pos - pos0);
            if (row < src_row0 || row - src_row0 >= src_rows) {
                continue;  // not this chip's rows (a local-rows source: the family gathers kg before the ring)
            }
            const uint32_t rt = (row - src_row0) / 32;
            if (rt != loaded) {
                for (uint32_t c = 0; c < KG_TILES; ++c) {
                    if constexpr (SRC_MODE == 1) {
                        const uint32_t x = get_arg_val<uint32_t>(ph_rt0 + 10 + 2 * rt),
                                       y = get_arg_val<uint32_t>(ph_rt0 + 11 + 2 * rt);
                        noc_async_read(
                            get_noc_addr(x, y, get_arg_val<uint32_t>(ph_rt0 + 2) + c * TB), kg_l1 + c * TB, TB);
                    } else {
                        noc_async_read_page(rt * KG_TILES + c, kg, kg_l1 + c * TB);
                    }
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                loaded = rt;
            }
            const uint32_t so = pmk::row_off_bf16(row % 32), dofs = pmk::row_off_bf16(j);
            for (uint32_t c = 0; c < KG_TILES; ++c) {
                copy32(kg_l1 + c * TB + so, scr + c * TB + dofs);
                copy32(kg_l1 + c * TB + so + 512, scr + c * TB + dofs + 512);
            }
        }
        invalidate_l1_cache();  // the RISC's stores land before the NoC reads the tiles
        for (uint32_t c = 0; c < RING_TILES; ++c) {
            noc_async_write_page(slot * RING_TILES + c, ring, scr + c * TB);
        }
        noc_async_write_barrier();
    }
}
