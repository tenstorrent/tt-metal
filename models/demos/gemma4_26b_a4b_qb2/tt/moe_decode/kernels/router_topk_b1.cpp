// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode batch-1 router tail on one core (data movement only): the E router logits of row 0 (FP32 TILE
// [1, 1, 1, E], E a multiple of 32) -> top-k, softmax over the k picks (equal to softmax over all E, then top-k,
// then renormalized), times per_expert_scale[id], written as the packed routing row RouterTopk.op_packed produces:
// [w_0 .. w_{k-1} | id_0 .. id_{k-1}] bf16 (ids exact below 256). Order: descending logit; equal logits keep the
// lower expert id first. Runs on FP32 logits, so no centering pass is needed for bf16 precision; the softmax is
// integer fixed-point arithmetic (the data-movement cores have no FPU).
// compile-time args: [cb_scr, num_experts, k] + TensorAccessorArgs(logits, pes, out)
// runtime args: [logits_addr, pes_addr, out_addr]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

// The cores have no FPU: everything below is integer arithmetic.

// order-preserving map of IEEE float bits to unsigned integers
inline uint32_t sortable(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

// IEEE float bits -> signed fixed point with 16 fractional bits (|value| < 2^15; smaller than 2^-16 -> 0)
inline int64_t to_q16(uint32_t bits) {
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF);
    if (exponent == 0) {
        return 0;
    }
    const int64_t mantissa = static_cast<int64_t>((bits & 0x7FFFFFu) | 0x800000u);  // value = m * 2^(e - 150)
    const int32_t shift = exponent - 150 + 16;
    int64_t q = shift >= 0 ? (mantissa << shift) : (shift > -40 ? (mantissa >> -shift) : 0);
    return (bits & 0x80000000u) ? -q : q;
}

// 2^y for y <= 0, y in Q16 -> Q30 (2^r on r in [0, 1) by a degree-5 polynomial in Q30, rel. error ~3e-7)
inline int64_t exp2_q30(int64_t y) {
    const int64_t n = y >> 16;        // floor
    const int64_t r = y - (n << 16);  // [0, 2^16)
    if (n <= -31) {
        return 0;
    }
    const int64_t rq = r << 14;  // Q30
    constexpr int64_t c[6] = {1073741824LL, 744261117LL, 257941803LL, 59603845LL, 10351951LL, 1355011LL};
    int64_t p = c[5];
    for (int i = 4; i >= 0; --i) {
        p = ((p * rq) >> 30) + c[i];
    }
    return p >> static_cast<int32_t>(-n);
}

// positive Q16 value -> bf16 bits, round to nearest even
inline uint16_t q16_to_bf16(int64_t q) {
    if (q <= 0) {
        return 0;
    }
    const int32_t msb = 63 - __builtin_clzll(static_cast<uint64_t>(q));
    const uint32_t exponent = static_cast<uint32_t>(msb - 16 + 127);
    // 24-bit mantissa with the hidden bit, then round to bf16's 8
    const uint64_t m24 = msb >= 23 ? static_cast<uint64_t>(q) >> (msb - 23) : static_cast<uint64_t>(q) << (23 - msb);
    const uint32_t bits = (exponent << 23) | static_cast<uint32_t>(m24 & 0x7FFFFFu);
    return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

inline uint16_t uint_to_bf16(uint32_t v) { return q16_to_bf16(static_cast<int64_t>(v) << 16); }

}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t pes_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t E = get_compile_time_arg_val(1);
    constexpr uint32_t k = get_compile_time_arg_val(2);
    constexpr auto l_args = TensorAccessorArgs<3>();
    constexpr auto p_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t f32_tile = 4096, f32_face = 1024, row16 = 64;  // FP32 tile, face, 16-value face row
    constexpr uint32_t tiles = E / 32;
    const auto s_l = TensorAccessor(l_args, logits_addr, f32_tile);
    const auto s_p = TensorAccessor(p_args, pes_addr, E * 2);
    const auto s_o = TensorAccessor(o_args, out_addr, 4 * k);

    // logits row 0: per tile, face 0 row 0 (columns 0-15) then face 1 row 0 (16-31); then the scale row
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint64_t src = s_l.get_noc_addr(t);
        noc_async_read(src, scr + t * 2 * row16, row16);
        noc_async_read(src + f32_face, scr + t * 2 * row16 + row16, row16);
    }
    const uint32_t pes_l1 = scr + E * 4;
    noc_async_read(s_p.get_noc_addr(0), pes_l1, E * 2);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* logit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    volatile tt_l1_ptr uint16_t* pes = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pes_l1);

    // one pass: keep the k largest (key, id) sorted descending; strict > keeps the lower id first on ties
    uint32_t key[k], id[k];
    for (uint32_t i = 0; i < k; ++i) {
        key[i] = 0;
        id[i] = 0;
    }
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t v = sortable(logit[e]);
        if (v <= key[k - 1]) {
            continue;
        }
        uint32_t pos = k - 1;
        while (pos > 0 && v > key[pos - 1]) {
            key[pos] = key[pos - 1];
            id[pos] = id[pos - 1];
            --pos;
        }
        key[pos] = v;
        id[pos] = e;
    }
    // softmax over the picks relative to the largest (Q16 logit differences, Q30 exponentials), times the
    // per-expert scale (bf16 -> Q16), normalized: Q16 weights
    constexpr int64_t log2e_q30 = 1549082005LL;  // log2(e) in Q30
    const int64_t top = to_q16(logit[id[0]]);
    int64_t e[k];
    int64_t sum = 0;
    for (uint32_t i = 0; i < k; ++i) {
        e[i] = exp2_q30(((to_q16(logit[id[i]]) - top) * log2e_q30) >> 30);
        sum += e[i];
    }
    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + E * 4 + E * 2);
    const int64_t inv_sum = (1LL << 60) / sum;  // Q30 sum -> Q30 reciprocal; one 64-bit division
    for (uint32_t i = 0; i < k; ++i) {
        const int64_t scale = to_q16(static_cast<uint32_t>(pes[id[i]]) << 16);
        const int64_t share = (e[i] * inv_sum) >> 30;  // Q30
        row[i] = q16_to_bf16((share * scale) >> 30);
        row[k + i] = uint_to_bf16(id[i]);
    }
    noc_async_write(scr + E * 4 + E * 2, s_o.get_noc_addr(0), 4 * k);
    noc_async_write_barrier();
}
