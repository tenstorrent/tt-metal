// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Router top-k, dataflow side (one core per token row, BRISC):
//   1. reads the row's E = 288 fp32 sigmoid scores (18 x 64 B face-row pieces of the 9 score tiles) and the
//      288 biases (row 0 of the bias tiles) into two fp32 tiles: column tile ct, face half h, element i lands
//      at tile word h * 256 + ct * 16 + i, i.e. the 64 B piece of tile ct / face half h is contiguous in L1;
//   2. hands both tiles to the compute kernel (choice = scores + bias, exact fp32) and takes the choice tile;
//   3. ranks the E choices with an insertion top-K on the fp32 bit patterns (monotonic unsigned key: sign bit
//      flipped, negative values complemented) -- strict comparisons, so equal values keep the LOWER column
//      first and a later equal value never displaces the current K-th (stable, lowest-index-first ties);
//   4. w_j = (s[idx_j] / sum_j s[idx_j]) * route_scale in soft fp32 (HF order, correctly rounded);
//   5. writes the row's pieces of the outputs: the dense [T,E] bf16 routing row (18 x 32 B, the weights at
//      their expert columns, 0 elsewhere), the w tile row (fp32 or bf16), the idx row in one of the formats
//      below, and (X_MODE = 1) the row-major [x | w] page the EP dispatch consumes (x untilized by the reader
//      kernel into CB_XROW, the K bf16 weights appended by this kernel). Rows >= T (padded token rows) write
//      zeros into the TILE outputs and nothing into the row-major pages (those have exactly T pages).
// Id formats (IDX_MODE): 1 uint32 tile row, 2 fp32 tile row, 3 uint16 ROW_MAJOR page (K x 2 B: the
// all_to_all_dispatch metadata), 4 bf16 tile row with IDX_PAD in columns K..31 (the single-chip EP op's index
// tiles). PERM_MODE = 1: the ids written to the idx outputs are perm[id] (a uint16 [E] page: the EP placement's
// global -> slot map, custom_ops/expert_stream/plan.py); the dense row keeps the global columns.
// Every L1 piece mirrors its DRAM offset modulo 64 (NoC alignment). Ends with noc_async_full_barrier().

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t CB_S = get_compile_time_arg_val(0);
constexpr uint32_t CB_B = get_compile_time_arg_val(1);
constexpr uint32_t CB_CH = get_compile_time_arg_val(2);
constexpr uint32_t CB_SCR = get_compile_time_arg_val(3);
constexpr uint32_t E = get_compile_time_arg_val(4);
constexpr uint32_t K = get_compile_time_arg_val(5);
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(6);
constexpr uint32_t W_MODE = get_compile_time_arg_val(7);          // 0 none, 1 fp32, 2 bf16
constexpr uint32_t IDX_MODE = get_compile_time_arg_val(8);        // 0 none, 1 uint32, 2 fp32, 3 uint16 rows, 4 bf16+pad
constexpr uint32_t DENSE_MODE = get_compile_time_arg_val(9);      // 0 none, 1 bf16
constexpr uint32_t X_MODE = get_compile_time_arg_val(10);         // 1: the [x | w] row-major page (reader + CB_XROW)
constexpr uint32_t PERM_MODE = get_compile_time_arg_val(11);      // 1: ids through the placement table
constexpr uint32_t CB_XROW = get_compile_time_arg_val(12);        // X_MODE: the untilized row + the w tail
constexpr uint32_t HT = get_compile_time_arg_val(13);             // X_MODE: H / 32
constexpr uint32_t XW_PAGE_BYTES = get_compile_time_arg_val(14);  // X_MODE: the page size of xw (2 (H + K))
constexpr uint32_t IDX_PAD_BITS = get_compile_time_arg_val(15);   // IDX_MODE 4: the bf16 pad id (bits)
constexpr auto scores_args = TensorAccessorArgs<16>();
constexpr auto bias_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
constexpr auto dense_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<dense_args.next_compile_time_args_offset()>();
constexpr auto idx_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto xw_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
constexpr auto perm_args = TensorAccessorArgs<xw_args.next_compile_time_args_offset()>();

static_assert(E % 32 == 0, "E must be a multiple of the tile width");
static_assert(K <= 16, "K must fit one face row");
constexpr uint32_t CT = E / 32;       // column tiles of the score row
constexpr uint32_t CHUNKS = CT * 2;   // 64 B face-row pieces per row
constexpr uint32_t FP32_FACE = 1024;  // bytes per 16x16 fp32 face
constexpr uint32_t BF16_FACE = 512;
// scratch layout (64 B aligned base): dense row pieces, then the w and idx face rows, then the perm page
constexpr uint32_t SCR_DENSE = 0;             // CHUNKS x 64 B slots (32 B used, mirrored offset)
constexpr uint32_t SCR_W = CHUNKS * 64;       // 128 B: fp32 face row pair, or 2 x 64 B bf16 slots
constexpr uint32_t SCR_IDX = SCR_W + 128;     // 128 B
constexpr uint32_t SCR_PERM = SCR_IDX + 128;  // E x 2 B (64 B aligned)
constexpr uint32_t PERM_BYTES = ((E * 2 + 63) / 64) * 64;
constexpr uint32_t SCR_BYTES = SCR_PERM + (PERM_MODE ? PERM_BYTES : 0);

FORCE_INLINE uint32_t fkey(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
FORCE_INLINE float as_float(uint32_t u) {
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
FORCE_INLINE uint32_t as_bits(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}
FORCE_INLINE uint32_t bf16_rne(uint32_t u) { return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
// word index of column c inside the row tile (column tile c / 32, face half (c % 32) / 16, element c % 16)
FORCE_INLINE uint32_t tile_word(uint32_t c) { return ((c & 31) >> 4) * 256 + (c >> 5) * 16 + (c & 15); }
}  // namespace

void kernel_main() {
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t dense_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t T = get_common_arg_val<uint32_t>(5);
    const uint32_t xw_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t perm_addr = get_common_arg_val<uint32_t>(7);
    const bool valid = row < T;

    const auto s_scores = TensorAccessor(scores_args, scores_addr);
    const auto s_bias = TensorAccessor(bias_args, bias_addr);
    const auto s_dense = TensorAccessor(dense_args, dense_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);
    const auto s_xw = TensorAccessor(xw_args, xw_addr);
    const auto s_perm = TensorAccessor(perm_args, perm_addr);

    CircularBuffer cbs(CB_S), cbb(CB_B), cbch(CB_CH), cbscr(CB_SCR);
    const uint32_t face = (row >> 4) * 2;  // first face of the row's half of a tile
    const uint32_t rf = row & 15;          // row within the face
    const uint32_t odd = (row & 1) * 32;   // bf16 face rows alternate the 64 B window halves

    // the staging area (reserved up front: the perm page is read with the scores)
    cbscr.reserve_back(1);
    const uint32_t scr = (cbscr.get_write_ptr() + 63u) & ~63u;

    // ---- 1. the score and bias tiles (+ the placement table)
    cbs.reserve_back(1);
    cbb.reserve_back(1);
    const uint32_t s_base = cbs.get_write_ptr();
    const uint32_t b_base = cbb.get_write_ptr();
    if (valid) {
        for (uint32_t ct = 0; ct < CT; ++ct) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t l1_off = h * FP32_FACE + ct * 64;
                noc_async_read(s_scores.get_noc_addr(ct, (face + h) * FP32_FACE + rf * 64), s_base + l1_off, 64);
                noc_async_read(s_bias.get_noc_addr(ct, h * FP32_FACE), b_base + l1_off, 64);
            }
        }
        if constexpr (PERM_MODE == 1) {
            noc_async_read(s_perm.get_noc_addr(0), scr + SCR_PERM, PERM_BYTES);
        }
        noc_async_read_barrier();
    }
    cbs.push_back(1);
    cbb.push_back(1);

    // ---- 2. the choice tile from the compute kernel
    cbch.wait_front(1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* ch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbch.get_read_ptr());
    volatile tt_l1_ptr uint32_t* sv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_base);

    // ---- 3. stable insertion top-K on the monotonic integer keys
    uint32_t bk[K], bi[K];
    uint32_t n = 0;
    if (valid) {
        for (uint32_t ct = 0; ct < CT; ++ct) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t wbase = h * 256 + ct * 16;
                const uint32_t cbase = ct * 32 + h * 16;
                for (uint32_t i = 0; i < 16; ++i) {
                    const uint32_t key = fkey(ch[wbase + i]);
                    uint32_t p;
                    if (n < K) {
                        p = n++;
                    } else if (key > bk[K - 1]) {
                        p = K - 1;
                    } else {
                        continue;
                    }
                    while (p > 0 && bk[p - 1] < key) {
                        bk[p] = bk[p - 1];
                        bi[p] = bi[p - 1];
                        --p;
                    }
                    bk[p] = key;
                    bi[p] = cbase + i;
                }
            }
        }
    }

    // ---- 4. the normalized weights (unbiased scores at the picks), HF order: (s / sum) * scale
    float wv[K];
    if (valid) {
        float sel[K];
        float den = 0.0f;
        for (uint32_t j = 0; j < K; ++j) {
            sel[j] = as_float(sv[tile_word(bi[j])]);
            den += sel[j];
        }
        const float scale = as_float(SCALE_BITS);
        for (uint32_t j = 0; j < K; ++j) {
            wv[j] = (sel[j] / den) * scale;
        }
    } else {
        for (uint32_t j = 0; j < K; ++j) {
            wv[j] = 0.0f;
            bi[j] = 0;
        }
    }

    // ---- 5. stage and write the row's output pieces
    volatile tt_l1_ptr uint32_t* scr_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    for (uint32_t i = 0; i < SCR_PERM / 4; ++i) {  // the output staging (not the perm page)
        scr_w[i] = 0;
    }
    if constexpr (DENSE_MODE == 1) {
        if (valid) {
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t c = bi[j];
                const uint32_t chunk = (c >> 5) * 2 + ((c & 31) >> 4);
                volatile tt_l1_ptr uint16_t* piece =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + SCR_DENSE + chunk * 64 + odd);
                piece[c & 15] = static_cast<uint16_t>(bf16_rne(as_bits(wv[j])));
            }
        }
        for (uint32_t ct = 0; ct < CT; ++ct) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(
                    scr + SCR_DENSE + (ct * 2 + h) * 64 + odd,
                    s_dense.get_noc_addr(ct, (face + h) * BF16_FACE + rf * 32),
                    32);
            }
        }
    }
    // the ids the idx outputs carry: the placement slots under PERM_MODE (the dense row above keeps the columns)
    uint32_t oi[K];
    for (uint32_t j = 0; j < K; ++j) {
        oi[j] = bi[j];
    }
    if constexpr (PERM_MODE == 1) {
        if (valid) {
            const volatile tt_l1_ptr uint16_t* perm =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(scr + SCR_PERM);
            for (uint32_t j = 0; j < K; ++j) {
                oi[j] = perm[bi[j]];
            }
        }
    }
    if constexpr (W_MODE == 1) {  // fp32 tile row: w in columns 0..K-1, zeros to column 31
        volatile tt_l1_ptr uint32_t* ww = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + SCR_W);
        for (uint32_t j = 0; j < K; ++j) {
            ww[j] = as_bits(wv[j]);
        }
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(scr + SCR_W + h * 64, s_w.get_noc_addr(0, (face + h) * FP32_FACE + rf * 64), 64);
        }
    } else if constexpr (W_MODE == 2) {  // bf16 tile row
        volatile tt_l1_ptr uint16_t* ww = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + SCR_W + odd);
        for (uint32_t j = 0; j < K; ++j) {
            ww[j] = static_cast<uint16_t>(bf16_rne(as_bits(wv[j])));
        }
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(scr + SCR_W + h * 64 + odd, s_w.get_noc_addr(0, (face + h) * BF16_FACE + rf * 32), 32);
        }
    }
    if constexpr (IDX_MODE == 1 || IDX_MODE == 2) {  // uint32 or fp32 (integral) tile row
        volatile tt_l1_ptr uint32_t* iw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + SCR_IDX);
        for (uint32_t j = 0; j < K; ++j) {
            iw[j] = (IDX_MODE == 1) ? oi[j] : as_bits(static_cast<float>(oi[j]));
        }
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(scr + SCR_IDX + h * 64, s_idx.get_noc_addr(0, (face + h) * FP32_FACE + rf * 64), 64);
        }
    } else if constexpr (IDX_MODE == 3) {  // uint16 ROW_MAJOR page (K x 2 B) of this token: rows >= T have no page
        if (valid) {
            volatile tt_l1_ptr uint16_t* iw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + SCR_IDX);
            for (uint32_t j = 0; j < K; ++j) {
                iw[j] = static_cast<uint16_t>(oi[j]);
            }
            noc_async_write(scr + SCR_IDX, s_idx.get_noc_addr(row), K * 2);
        }
    } else if constexpr (IDX_MODE == 4) {  // bf16 tile row: the ids as exact bf16 integers, IDX_PAD beyond K
        volatile tt_l1_ptr uint16_t* iw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + SCR_IDX + odd);
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t v = j < K ? bf16_rne(as_bits(static_cast<float>(oi[j]))) : IDX_PAD_BITS;
            iw[j < 16 ? j : 32 + (j - 16)] = static_cast<uint16_t>(v);  // the second face half 64 B further
        }
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(scr + SCR_IDX + h * 64 + odd, s_idx.get_noc_addr(0, (face + h) * BF16_FACE + rf * 32), 32);
        }
    }
    if constexpr (X_MODE == 1) {  // the [x | w] row-major page: the reader's untilized row + the bf16 weights
        CircularBuffer cbrow(CB_XROW);
        cbrow.wait_front(1);
        if (valid) {
            volatile tt_l1_ptr uint16_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbrow.get_read_ptr() + HT * 64);
            for (uint32_t j = 0; j < K; ++j) {
                tail[j] = static_cast<uint16_t>(bf16_rne(as_bits(wv[j])));
            }
            noc_async_write(cbrow.get_read_ptr(), s_xw.get_noc_addr(row), XW_PAGE_BYTES);
        }
        noc_async_write_barrier();
        cbrow.pop_front(1);
    }
    noc_async_write_barrier();
    cbch.pop_front(1);
    noc_async_full_barrier();
}
