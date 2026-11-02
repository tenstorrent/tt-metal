// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV41_KVSTEP_SPLIT=<NPART> (default off): paged_kv_step.cpp with every query row served by NPART cores instead of one
// (core id = row * NPART + part). The single-core kernel is RISC-bound (software bf16 -> fp8 conversion of the
// 512-element ring row + a 640-entry index build, 29.5 us per row at 8k); here each part
//   * converts and writes 1/NPART of the ring row (chunks [part * 16 / NPART, (part + 1) * 16 / NPART) of the 16 x
//   32-element chunks, i.e. a 512 / NPART byte slice of the fp8 pool row),
//   * the same for the compressed latent row (HAS_LAT),
//   * builds a contiguous aligned slice [lo, hi) of the TOPK index row from a closed form: o < nr: ring row of the
//   window, else the entry o - nr of the selection translated through the page table, else the
//     0xFFFFFFFF sentinel. This equals the compaction loop of the single-core kernel whenever no selected entry is
//     skipped in the middle (the selection is sorted valid entries followed by a sentinel tail, the precondition of
//     sparse_sdpa itself); bit-identical rows are checked by tests/test_fusedec_kvstep.py.
// Compile-time args as paged_kv_step.cpp plus NPART at index 15 (accessors start at 16). Pool rows are written in
// slices; the fp8 pool is the supported split format (a bf16 pool uses the same slicing without conversion).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline uint32_t bf16_to_e4m3(uint32_t b) {
    const uint32_t sign = (b >> 8) & 0x80;
    const uint32_t e = (b >> 7) & 0xFF, m = b & 0x7F;
    if (e == 0xFF) {
        return sign | ((m != 0) ? 0x7F : 0x7E);
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
    const int32_t sh = 125 - (int32_t)e;
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

// dest words [i0, i1) of the fp8 row (4 e4m3 bytes per word) from source words [2 * i0, 2 * i1) of the bf16 row, in
// place (disjoint regions for i0 > 0, ascending for i0 = 0)
static inline void part_to_fp8(volatile tt_l1_ptr uint32_t* w, uint32_t i0, uint32_t i1) {
    for (uint32_t i = i0; i < i1; ++i) {
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
    constexpr uint32_t RATIO = get_compile_time_arg_val(4);
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
    constexpr uint32_t NPART = get_compile_time_arg_val(15);
    constexpr uint32_t ROW_B = POOL_FP8 ? 512 : 1024;
    constexpr auto kv_args = TensorAccessorArgs<16>();
    constexpr auto lat_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto pool_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();

    constexpr uint32_t NCOMP = TOPK - WIN;
    constexpr uint32_t CH = 16 / NPART;                           // 32-element chunks of the row per part
    constexpr uint32_t SLICE_B = ROW_B / NPART;                   // bytes of the pool row per part
    constexpr uint32_t SLICE_W = (POOL_FP8 ? 128 : 256) / NPART;  // destination words per part
    constexpr uint32_t OUT_PER = TOPK / NPART;                    // index entries per part (multiple of 16)
    constexpr uint32_t POS_B = ((ROWS * 4 + 63) / 64) * 64;
    constexpr uint32_t B_POS = 0, B_KV = B_POS + POS_B, B_LAT = B_KV + 16 * 1024, B_PT = B_LAT + 16 * 1024,
                       B_IDS = B_PT + ((MAXP * 4 + 63) / 64) * 64, B_ROW = B_IDS + 2048, B_ROW2 = B_ROW + 1024,
                       B_OUT = B_ROW2 + 1024;

    const uint32_t id = get_arg_val<uint32_t>(0);
    const uint32_t row = id / NPART, part = id % NPART;
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

    noc.async_read(pos_acc, cb, ROWS * 4, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_POS});
    if constexpr (RATIO > 0) {
        noc.async_read(pt_acc, cb, MAXP * 4, {.page_id = user, .offset_bytes = 0}, {.offset_bytes = B_PT});
    }
    [[maybe_unused]] const uint32_t kv_r = (KV_MODE == 0) ? 0 : (row % 32);
    [[maybe_unused]] const uint32_t kv_page0 = (KV_MODE == 0) ? row * 16 : (row / 32) * 16;
    [[maybe_unused]] const uint32_t kv_off = (kv_r >= 16) ? 1024 : 0;
    if constexpr (WRITE_KV) {
        for (uint32_t c = part * CH; c < (part + 1) * CH; ++c) {
            noc.async_read(
                kv_acc, cb, 1024, {.page_id = kv_page0 + c, .offset_bytes = kv_off}, {.offset_bytes = B_KV + c * 1024});
        }
    }
    [[maybe_unused]] const uint32_t lat_r = row % 32;
    if constexpr (HAS_LAT) {
        const uint32_t lat_off = (lat_r >= 16) ? 1024 : 0;
        for (uint32_t c = part * CH; c < (part + 1) * CH; ++c) {
            noc.async_read(
                lat_acc,
                cb,
                1024,
                {.page_id = (row / 32) * 16 + c, .offset_bytes = lat_off},
                {.offset_bytes = B_LAT + c * 1024});
        }
    }
    constexpr bool NEED_IDS = RATIO > 0;
    // ids row (only read when the selection is the indexer's: N > NCOMP, decided after pos is known)
    noc.async_read_barrier();

    const int32_t pos = posb[row];
    const bool active = pos >= 0;
    const uint32_t p = active ? (uint32_t)pos : 0;
    const uint32_t pos_u = p;
    const uint32_t ring_user = ring_base + user * RING;

    uint32_t N = 0;
    if constexpr (NEED_IDS) {
        N = (p + 1) / RATIO;
        if (N > NCOMP) {
            noc.async_read(ids_acc, cb, 2048, {.page_id = row, .offset_bytes = 0}, {.offset_bytes = B_IDS});
        }
    }

    // ---- 1. slice of the ring row
    if constexpr (WRITE_KV) {
        if (active) {
            const uint32_t rr = kv_r & 15;
            for (uint32_t c = part * CH; c < (part + 1) * CH; ++c) {
                for (uint32_t w = 0; w < 8; ++w) {
                    rowb[c * 16 + w] = kvb[c * 256 + rr * 8 + w];
                    rowb[c * 16 + 8 + w] = kvb[c * 256 + 128 + rr * 8 + w];
                }
            }
            if constexpr (POOL_FP8) {
                part_to_fp8(rowb, part * SLICE_W, (part + 1) * SLICE_W);
            }
            noc.async_write(
                cb,
                pool_acc,
                SLICE_B,
                {.offset_bytes = B_ROW + part * SLICE_B},
                {.page_id = ring_user + (p % RING), .offset_bytes = part * SLICE_B});
        }
    }
    // ---- 2. slice of the compressed latent row (the group-completing position only)
    if constexpr (HAS_LAT) {
        if (active && ((pos_u + 1) % RATIO == 0)) {
            const uint32_t rr = lat_r & 15;
            for (uint32_t c = part * CH; c < (part + 1) * CH; ++c) {
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
                    part_to_fp8(row2b, part * SLICE_W, (part + 1) * SLICE_W);
                }
                noc.async_write(
                    cb,
                    pool_acc,
                    SLICE_B,
                    {.offset_bytes = B_ROW2 + part * SLICE_B},
                    {.page_id = dst, .offset_bytes = part * SLICE_B});
            }
        }
    }

    // ---- 3. this part's slice of the index row
    if constexpr (NEED_IDS) {
        if (N > NCOMP) {
            noc.async_read_barrier();
        }
    }
    const uint32_t nr = (p + 1 < WIN) ? (p + 1) : WIN;
    const uint32_t lo = part * OUT_PER, hi = lo + OUT_PER;
    uint32_t cnt = 0;
    if constexpr (NEED_IDS) {
        cnt = (N > NCOMP) ? NCOMP : N;
    }
    for (uint32_t o = lo; o < hi; ++o) {
        uint32_t v = 0xFFFFFFFFu;
        if (o < nr) {
            const uint32_t q = p - (nr - 1) + o;
            v = ring_user + (q % RING);
        } else if constexpr (NEED_IDS) {
            const uint32_t i = o - nr;
            if (i < cnt) {
                const uint32_t e = (N > NCOMP) ? idsb[i] : i;
                if (e < N) {
                    const uint32_t tok = e * RATIO;
                    const int32_t page = ptb[tok / PAGE_TOK];
                    if (page >= 0) {
                        v = (uint32_t)page * PAGE_ROWS + SRC_OFF + (tok % PAGE_TOK) / RATIO;
                    }
                }
            }
        }
        outb[o] = v;
    }
    noc.async_write(
        cb, out_acc, OUT_PER * 4, {.offset_bytes = B_OUT + lo * 4}, {.page_id = row, .offset_bytes = lo * 4});
    noc.async_write_barrier();
}
