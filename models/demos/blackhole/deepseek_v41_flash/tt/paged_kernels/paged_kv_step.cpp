// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash paged decode KV step, one data-movement kernel per query row (one Tensix core per row).
//
// For query row i (user = i / NQ, position pos[i]) of ONE attention layer this op
//   1. writes the layer's new window K==V row into the layer's RING region of the row-major pool at slot pos % RING,
//   2. (HAS_LAT) writes the new compressed latent into the shared page pool at the page-table translated row of
//      entry pos / RATIO (the entry of the group in progress: invisible to every reader until the group completes),
//   3. builds the sparse_sdpa index row [TOPK] uint32 of this query: the valid ring rows, the selected compressed
//      rows (all entries when fewer than TOPK - WIN exist, else the indexer's ids < N), then 0xFFFFFFFF sentinels
//      (contiguous tail, at least one valid row).
//
//   pool     bf16 or fp8_e4m3 (POOL_FP8) [1,1,R,512] ROW_MAJOR (page = 1 row = 1024 / 512 B; rows are converted here
//   with RNE + saturation). Rows [0, NP*PAGE_ROWS) are the shared pages,
//            [ring_base, ..) the static rings: ring row of (user u, slot s) = ring_base + u * RING + s.
//   kv       bf16 TILE. KV_MODE 0: [1,T,32,512] (user t = tile row-block t, kv vector = row 0, the q-tile layout of the
//            attention); KV_MODE 1: [1,1,rows,512] (row i of the tile rows).
//   lat      bf16 TILE [1,1,rows,512], row i = latent of query row i (RoPE'd).
//   pos      int32 ROW_MAJOR [rows] (one page): position of every row (< 0: row inactive, no write, 1 dummy index).
//   pt       int32 ROW_MAJOR [users, MAXP]: page table, page id of logical page p (128 tokens) of every user, -1 free.
//   ids      uint32 ROW_MAJOR [rows, 512]: indexer top-k entry ids (0xFFFFFFFF = none), only read when N > TOPK - WIN.
//   out      uint32 ROW_MAJOR [1,1,rows,TOPK]
// Entry j of a source with ratio RATIO lives at pool row pt[u][(j*RATIO)/PAGE_TOK] * PAGE_ROWS + SRC_OFF + ((j*RATIO) %
// PAGE_TOK) / RATIO.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// bf16 -> fp8 e4m3fn (sign, 4-bit exponent bias 7, 3-bit mantissa; round to nearest even, saturating at 448, no inf)
static inline uint32_t bf16_to_e4m3(uint32_t b) {
    const uint32_t sign = (b >> 8) & 0x80;
    const uint32_t e = (b >> 7) & 0xFF, m = b & 0x7F;
    if (e == 0xFF) {
        return sign | ((m != 0) ? 0x7F : 0x7E);  // NaN stays NaN, inf saturates
    }
    if (e == 0) {
        return sign;
    }
    const int32_t ef = (int32_t)e - 127 + 7;
    if (ef >= 1) {
        const uint32_t mant = m >> 4, rem = m & 0xF;
        uint32_t v = ((uint32_t)ef << 3) | mant;
        if (rem > 8 || (rem == 8 && (mant & 1))) {
            v += 1;
        }
        return sign | (v > 0x7E ? 0x7E : v);
    }
    const int32_t sh = 125 - (int32_t)e;  // subnormal: units of 2^-9
    if (sh > 9) {
        return sign;
    }
    const uint32_t M = 128 | m;
    uint32_t v = M >> sh;
    const uint32_t rem = M & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (v & 1))) {
        v += 1;
    }
    return sign | v;
}

// 512 bf16 (uint16 pairs in 256 words) -> 512 e4m3 bytes (128 words), in place at ``w``
static inline void row_to_fp8(volatile tt_l1_ptr uint32_t* w) {
    for (uint32_t i = 0; i < 128; ++i) {
        const uint32_t a = w[2 * i], b = w[2 * i + 1];
        w[i] = bf16_to_e4m3(a & 0xFFFF) | (bf16_to_e4m3(a >> 16) << 8) | (bf16_to_e4m3(b & 0xFFFF) << 16) |
               (bf16_to_e4m3(b >> 16) << 24);
    }
}

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t NQ = get_compile_time_arg_val(1);
    constexpr uint32_t RING = get_compile_time_arg_val(2);
    constexpr uint32_t WIN = get_compile_time_arg_val(3);
    constexpr uint32_t RATIO = get_compile_time_arg_val(4);  // 0: window-only layer
    constexpr uint32_t SRC_OFF = get_compile_time_arg_val(5);
    constexpr uint32_t PAGE_TOK = get_compile_time_arg_val(6);
    constexpr uint32_t PAGE_ROWS = get_compile_time_arg_val(7);
    constexpr uint32_t TOPK = get_compile_time_arg_val(8);
    constexpr uint32_t MAXP = get_compile_time_arg_val(9);
    constexpr uint32_t HAS_LAT = get_compile_time_arg_val(10);
    constexpr uint32_t KV_MODE = get_compile_time_arg_val(11);
    constexpr uint32_t WRITE_KV = get_compile_time_arg_val(12);
    constexpr uint32_t cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t POOL_FP8 = get_compile_time_arg_val(14);
    constexpr uint32_t ROW_B = POOL_FP8 ? 512 : 1024;  // pool page = one row
    constexpr auto kv_args = TensorAccessorArgs<15>();
    constexpr auto lat_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto pool_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();

    constexpr uint32_t NCOMP = TOPK - WIN;  // selected compressed rows
    constexpr uint32_t POS_B = ((ROWS * 4 + 63) / 64) * 64;
    constexpr uint32_t B_POS = 0, B_KV = B_POS + POS_B, B_LAT = B_KV + 16 * 1024, B_PT = B_LAT + 16 * 1024,
                       B_IDS = B_PT + ((MAXP * 4 + 63) / 64) * 64, B_ROW = B_IDS + 2048, B_ROW2 = B_ROW + 1024,
                       B_OUT = B_ROW2 + 1024;

    const uint32_t row = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t a_kv = get_common_arg_val<uint32_t>(0), a_lat = get_common_arg_val<uint32_t>(1),
                                    a_pos = get_common_arg_val<uint32_t>(2), a_pt = get_common_arg_val<uint32_t>(3),
                                    a_ids = get_common_arg_val<uint32_t>(4), a_pool = get_common_arg_val<uint32_t>(5),
                                    a_out = get_common_arg_val<uint32_t>(6),
                                    ring_base = get_common_arg_val<uint32_t>(7);

    Noc noc;
    const auto kv_acc = TensorAccessor(kv_args, a_kv, 2048);
    const auto lat_acc = TensorAccessor(lat_args, a_lat, 2048);
    const auto pos_acc = TensorAccessor(pos_args, a_pos, ROWS * 4);
    const auto pt_acc = TensorAccessor(pt_args, a_pt, MAXP * 4);
    const auto ids_acc = TensorAccessor(ids_args, a_ids, 2048);
    const auto pool_acc = TensorAccessor(pool_args, a_pool, ROW_B);
    const auto out_acc = TensorAccessor(out_args, a_out, TOPK * 4);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr int32_t* posb = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_POS);
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* kvb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_KV);
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* latb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_LAT);
    [[maybe_unused]] volatile tt_l1_ptr int32_t* ptb = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_PT);
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* idsb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_IDS);
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* rowb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_ROW);
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* row2b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_ROW2);
    volatile tt_l1_ptr uint32_t* outb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_OUT);

    const uint32_t user = row / NQ;

    // ---- loads: positions, page table row of the user, new rows (first two faces of the tiles that hold them)
    noc.async_read(pos_acc, cb, ROWS * 4, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_POS});
    if constexpr (RATIO > 0) {
        noc.async_read(pt_acc, cb, MAXP * 4, {.page_id = user, .offset_bytes = 0}, {.offset_bytes = B_PT});
    }
    // kv row: KV_MODE 0: tile page (user*16 + c), row 0 -> faces 0/1 rows 0; KV_MODE 1: row i of the [rows] tile rows
    [[maybe_unused]] const uint32_t kv_r = (KV_MODE == 0) ? 0 : (row % 32);
    [[maybe_unused]] const uint32_t kv_page0 = (KV_MODE == 0) ? user * 16 : (row / 32) * 16;
    [[maybe_unused]] const uint32_t kv_off = (kv_r >= 16) ? 1024 : 0;
    if constexpr (WRITE_KV) {
        for (uint32_t c = 0; c < 16; ++c) {
            noc.async_read(
                kv_acc, cb, 1024, {.page_id = kv_page0 + c, .offset_bytes = kv_off}, {.offset_bytes = B_KV + c * 1024});
        }
    }
    [[maybe_unused]] const uint32_t lat_r = row % 32;
    if constexpr (HAS_LAT) {
        const uint32_t lat_off = (lat_r >= 16) ? 1024 : 0;
        for (uint32_t c = 0; c < 16; ++c) {
            noc.async_read(
                lat_acc,
                cb,
                1024,
                {.page_id = (row / 32) * 16 + c, .offset_bytes = lat_off},
                {.offset_bytes = B_LAT + c * 1024});
        }
    }
    noc.async_read_barrier();

    const int32_t pos = posb[row];
    const bool active = pos >= 0;
    const uint32_t p = active ? (uint32_t)pos : 0;
    const uint32_t ring_user = ring_base + user * RING;

    // ---- 1. ring row write
    if constexpr (WRITE_KV) {
        if (active) {
            const uint32_t rr = kv_r & 15;
            for (uint32_t c = 0; c < 16; ++c) {
                for (uint32_t w = 0; w < 8; ++w) {
                    rowb[c * 16 + w] = kvb[c * 256 + rr * 8 + w];
                    rowb[c * 16 + 8 + w] = kvb[c * 256 + 128 + rr * 8 + w];
                }
            }
            if constexpr (POOL_FP8) {
                row_to_fp8(rowb);
            }
            noc.async_write(
                cb, pool_acc, ROW_B, {.offset_bytes = B_ROW}, {.page_id = ring_user + (p % RING), .offset_bytes = 0});
        }
    }
    // ---- 2. compressed latent write
    if constexpr (HAS_LAT) {
        if (active) {
            const uint32_t rr = lat_r & 15;
            for (uint32_t c = 0; c < 16; ++c) {
                for (uint32_t w = 0; w < 8; ++w) {
                    row2b[c * 16 + w] = latb[c * 256 + rr * 8 + w];
                    row2b[c * 16 + 8 + w] = latb[c * 256 + 128 + rr * 8 + w];
                }
            }
            const uint32_t j = p / RATIO;
            const uint32_t tok = j * RATIO;
            const int32_t page = ptb[tok / PAGE_TOK];
            if (page >= 0) {
                const uint32_t dst = (uint32_t)page * PAGE_ROWS + SRC_OFF + (tok % PAGE_TOK) / RATIO;
                if constexpr (POOL_FP8) {
                    row_to_fp8(row2b);
                }
                noc.async_write(cb, pool_acc, ROW_B, {.offset_bytes = B_ROW2}, {.page_id = dst, .offset_bytes = 0});
            }
        }
    }

    // ---- 3. indices
    uint32_t n = 0;
    const uint32_t nr = (p + 1 < WIN) ? (p + 1) : WIN;
    for (uint32_t i = 0; i < nr; ++i) {
        const uint32_t q = p - (nr - 1) + i;
        outb[n++] = ring_user + (q % RING);
    }
    if constexpr (RATIO > 0) {
        const uint32_t N = (p + 1) / RATIO;
        if (N > NCOMP) {
            noc.async_read(ids_acc, cb, 2048, {.page_id = row, .offset_bytes = 0}, {.offset_bytes = B_IDS});
            noc.async_read_barrier();
        }
        const uint32_t cnt = (N > NCOMP) ? NCOMP : N;
        for (uint32_t i = 0; i < cnt; ++i) {
            const uint32_t e = (N > NCOMP) ? idsb[i] : i;
            if (e >= N) {
                continue;  // sentinel / not yet visible
            }
            const uint32_t tok = e * RATIO;
            const int32_t page = ptb[tok / PAGE_TOK];
            if (page < 0) {
                continue;
            }
            outb[n++] = (uint32_t)page * PAGE_ROWS + SRC_OFF + (tok % PAGE_TOK) / RATIO;
        }
    }
    if (n == 0) {
        outb[n++] = ring_user;  // inactive row: any valid row keeps the kernel's "at least one valid key" precondition
    }
    for (uint32_t i = n; i < TOPK; ++i) {
        outb[i] = 0xFFFFFFFFu;
    }
    noc.async_write(cb, out_acc, TOPK * 4, {.offset_bytes = B_OUT}, {.page_id = row, .offset_bytes = 0});
    noc.async_write_barrier();
}
