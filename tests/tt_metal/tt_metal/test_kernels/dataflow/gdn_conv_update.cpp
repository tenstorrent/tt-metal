// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Spike E (GDN decode v2): the causal-conv state update on one data-movement RISC, in software fp32.
//   state  [slots, W-1, D] bf16 row-major in DRAM: the last W-1 inputs of each slot (oldest row first)
//   x      [T, D]          bf16 row-major: this call's tokens
//   weight [D, W]          bf16 row-major: taps, oldest first (w[d][W-1] multiplies the current token)
//   bias   [D]             bf16
//   y      [T, D]          bf16 row-major: silu(sum_i w[d][i] * window_i[d] + bias[d]) per token
// For token t the window is the W-1 previous inputs followed by x_t; after the call the slot holds the last W-1 inputs
// of [state; x_0..x_{T-1}]. Every token runs the same code, so one call of T tokens equals T chained calls.
// The slot index comes from a one-row int32 index buffer (a negative index skips the row untouched).
// Runtime args: state addr, x addr, weight addr, bias addr, y addr, index addr, scratch L1 addr, T.
// Compile-time args: D, W. bf16 <-> fp32 by bit shifts (RNE on the way back), exp by range reduction + polynomial.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t D = get_compile_time_arg_val(0);
constexpr uint32_t W = get_compile_time_arg_val(1);
constexpr uint32_t ROW_BYTES = D * 2;

inline float bf16_to_f32(uint16_t b) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = static_cast<uint32_t>(b) << 16;
    return c.f;
}

inline uint16_t f32_to_bf16_rne(float f) {
    union {
        uint32_t u;
        float f;
    } c;
    c.f = f;
    const uint32_t lsb = (c.u >> 16) & 1u;
    const uint32_t rounding = 0x7FFFu + lsb;
    return static_cast<uint16_t>((c.u + rounding) >> 16);
}

// exp(x) for x in a sane range: 2^k * p(r), x = k ln2 + r, |r| <= ln2 / 2, degree-6 Taylor of e^r.
inline float exp_f32(float x) {
    if (x > 88.0f) {
        x = 88.0f;
    }
    if (x < -87.0f) {
        return 0.0f;
    }
    const float log2e = 1.44269504f, ln2_hi = 0.693145751953125f, ln2_lo = 1.42860677e-6f;
    const int k = static_cast<int>(x * log2e + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = (x - static_cast<float>(k) * ln2_hi) - static_cast<float>(k) * ln2_lo;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    union {
        uint32_t u;
        float f;
    } s;
    s.u = static_cast<uint32_t>(k + 127) << 23;
    return p * s.f;
}

inline float silu(float v) { return v / (1.0f + exp_f32(-v)); }
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    const uint32_t y_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_addr = get_arg_val<uint32_t>(5);
    const uint32_t scratch = get_arg_val<uint32_t>(6);
    const uint32_t T = get_arg_val<uint32_t>(7);
    constexpr uint32_t bank = 0;

    // scratch layout: window rows (W-1+T) x D bf16, then weight D x W bf16, bias D bf16, y T x D bf16, index 32 B
    const uint32_t win_l1 = scratch;
    const uint32_t w_l1 = win_l1 + (W - 1 + T) * ROW_BYTES;
    const uint32_t b_l1 = w_l1 + D * W * 2;
    const uint32_t y_l1 = b_l1 + ROW_BYTES;
    const uint32_t idx_l1 = y_l1 + T * ROW_BYTES;

    const uint64_t idx_noc = get_noc_addr_from_bank_id<true>(bank, idx_addr);
    noc_async_read(idx_noc, idx_l1, 32);
    noc_async_read_barrier();
    const int32_t slot = *reinterpret_cast<volatile tt_l1_ptr int32_t*>(idx_l1);
    if (slot < 0) {
        return;  // a skipped row: its state and its y rows stay untouched
    }
    const uint32_t slot_bytes = (W - 1) * ROW_BYTES;
    noc_async_read(get_noc_addr_from_bank_id<true>(bank, state_addr + slot * slot_bytes), win_l1, slot_bytes);
    noc_async_read(get_noc_addr_from_bank_id<true>(bank, x_addr), win_l1 + slot_bytes, T * ROW_BYTES);
    noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr), w_l1, D * W * 2);
    noc_async_read(get_noc_addr_from_bank_id<true>(bank, b_addr), b_l1, ROW_BYTES);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* win = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(win_l1);
    volatile tt_l1_ptr uint16_t* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1);
    volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
    volatile tt_l1_ptr uint16_t* y = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(y_l1);
    for (uint32_t t = 0; t < T; t++) {
        // the window of token t is rows t .. t+W-1 of [state; x]
        for (uint32_t d = 0; d < D; d++) {
            float acc = bf16_to_f32(b[d]);
            for (uint32_t i = 0; i < W; i++) {
                acc += bf16_to_f32(w[d * W + i]) * bf16_to_f32(win[(t + i) * D + d]);
            }
            y[t * D + d] = f32_to_bf16_rne(silu(acc));
        }
    }
    // y out; the new state is the last W-1 rows of [state; x], i.e. rows T .. T+W-2 of the window
    noc_async_write(y_l1, get_noc_addr_from_bank_id<true>(bank, y_addr), T * ROW_BYTES);
    noc_async_write(
        win_l1 + T * ROW_BYTES, get_noc_addr_from_bank_id<true>(bank, state_addr + slot * slot_bytes), slot_bytes);
    noc_async_write_barrier();
}
