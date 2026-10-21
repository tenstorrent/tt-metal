// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Top-k router selection of the DeepSeek-V4.1 decode step (tt/v41/routing.py router_select): one core per token row
// t, exact fp32 top-K of rank[t, :E] (the lowest index wins a tie; ``moe_grouped_topk`` sorts TF32 keys, 13 mantissa
// bits fewer, and merges its per-tile results by chain position, so the two differ on near-ties, where the exact
// one is the reference's), then
// w_j = bf16(score[t, idx_j] / (sum_j score[t, idx_j] + eps) * scale) with round-to-nearest-even. ``rank`` and
// ``score`` are fp32 TILE tensors [T, E] (T a multiple of 32 or below it; E / 32 tiles across); ``rank`` is score +
// bias. Rows at or past REAL_ROWS are the padding of the decode tile: they get the sentinel index (E) and weight 0, as
// ``moe_grouped_topk`` marks them under a padding config, so the dispatch, combine and reduce skip them.
//
// Outputs: ``w`` row-major bf16 [T, K] (one page per row of K * 2 bytes) and ``idx`` uint16 TILE [T, K] (32 x 32
// tiles; row t's K indices land at its row of face (t / 16) * 2 of tile t / 32: 16-element faces, row-major within).
//
// After models/demos/blackhole/deepseek_v41_flash/tt/mhc_kernels/router_select.cpp (ssinghal/dsv4p1), with the
// padding sentinel and the tiled index output for the decode step's routing setup (kernels/route_offsets.cpp).
//
// compile_time_args = [cb_rank, cb_score, cb_out, E / 32, K, eps bits, scale bits, REAL_ROWS,
//                      TensorAccessorArgs(rank)..., (score)..., (w)..., (idx)...]
// runtime args      = [rank_addr, score_addr, w_addr, idx_addr, t]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

static inline uint32_t fkey(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
static inline float as_f(uint32_t b) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = b;
    return c.f;
}
static inline uint32_t as_u(float f) {
    union {
        uint32_t u;
        float f;
    } c;
    c.f = f;
    return c.u;
}

void kernel_main() {
    const uint32_t rank_addr = get_arg_val<uint32_t>(0);
    const uint32_t score_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t t = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_rank = get_compile_time_arg_val(0);
    constexpr uint32_t cb_score = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);  // E / 32
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t REAL_ROWS = get_compile_time_arg_val(7);
    constexpr auto rank_args = TensorAccessorArgs<8>();
    constexpr auto score_args = TensorAccessorArgs<rank_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<score_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t E = NT * 32;

    // page strides come from the accessor args (the buffers' aligned page sizes: 4 KB fp32 tiles, 2 KB uint16
    // tiles, the aligned K * 2 bytes of a weights row)
    const auto rank = TensorAccessor(rank_args, rank_addr);
    const auto score = TensorAccessor(score_args, score_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    DataflowBuffer rank_buf(cb_rank), score_buf(cb_score), out_buf(cb_out);
    const uint32_t rank_l1 = rank_buf.get_write_ptr();
    const uint32_t score_l1 = score_buf.get_write_ptr();
    const uint32_t out_l1 = out_buf.get_write_ptr();
    volatile tt_l1_ptr uint16_t* ow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1);
    volatile tt_l1_ptr uint16_t* oi = ow + 32;  // +64 B: the index row

    // the row inside its (only) tile row: fp32 faces are 16 x 16 x 4 B = 1 KB, two across per tile
    const uint32_t tr = t & 31;
    const uint32_t pbase = (t >> 5) * NT;  // the first tile page of the row's tile row
    const uint32_t roff = ((tr >> 4) << 11) + ((tr & 15) << 6);
    uint32_t bi[K];
    if (t < REAL_ROWS) {
        for (uint32_t n = 0; n < NT; ++n) {
            noc_async_read(rank.get_noc_addr(pbase + n, roff), rank_l1 + n * 128, 64);
            noc_async_read(rank.get_noc_addr(pbase + n, roff + 1024), rank_l1 + n * 128 + 64, 64);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* rk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rank_l1);
        uint32_t bk[K];
        for (uint32_t j = 0; j < K; ++j) {
            bk[j] = 0;
            bi[j] = 0;
        }
        // strictly-greater insertion in ascending index order: the lowest index wins a tie
        for (uint32_t c = 0; c < E; ++c) {
            const uint32_t key = fkey(rk[c]);
            if (key > bk[K - 1]) {
                uint32_t p = K - 1;
                while (p > 0 && key > bk[p - 1]) {
                    bk[p] = bk[p - 1];
                    bi[p] = bi[p - 1];
                    --p;
                }
                bk[p] = key;
                bi[p] = c;
            }
        }
        // the unbiased scores at the chosen experts: the 16-element half row holding each
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t c = bi[j];
            noc_async_read(score.get_noc_addr(pbase + (c >> 5), roff + (((c >> 4) & 1) << 10)), score_l1 + j * 64, 64);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(score_l1);
        float s[K];
        float den = 0.0f;
        for (uint32_t j = 0; j < K; ++j) {
            s[j] = as_f(sc[j * 16 + (bi[j] & 15)]);
            den += s[j];
        }
        const float inv = as_f(EPS_BITS) + den;
        const float scale = as_f(SCALE_BITS);
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t b = as_u((s[j] / inv) * scale);
            ow[j] = (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
            oi[j] = (uint16_t)bi[j];
        }
    } else {
        for (uint32_t j = 0; j < K; ++j) {
            ow[j] = 0;
            oi[j] = (uint16_t)E;
        }
    }
    noc_async_write(out_l1, w.get_noc_addr(t, 0), K * 2);
    noc_async_write(out_l1 + 64, idx.get_noc_addr(t >> 5, ((tr >> 4) << 10) + ((tr & 15) << 5)), K * 2);
    noc_async_write_barrier();
}
