// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_SPEC_CHAIN_KERNEL (tt/spec_fused_kernels.py): the S3c chain trace body (spec_chain.ChainDevice.body, 126 ttnn ops
// of int32 glue and fp32 TILE elementwise math) as ONE single-core data-movement kernel. Same buffers, same words.
//
// Inputs (ROW_MAJOR, one page per row):
//   result_in   int32  [2 or 3, W_RES]  row 0 committed[R] (-1 past the count), row 1 counts[B]; row 2 is not read
//   aux_prev_in int32  [7, W_A]         row AUX_P0 = p0_prev[B] (read BEFORE this kernel overwrites the tensor)
//   draft k     uint32 [1,1,B,1]        K tensors, one id per page (the drafter's draft_argmax outputs)
//   ovr_in      int32  [11, W_O]        the host overlay (spec_chain.pack_ovr_host layout)
// Outputs (every word of every page is written, zero padding included, as the op graph's ttnn.copy does):
//   tok_out     uint32 [1, R]           next verify token block
//   vaux_out    uint32 [3, W_V]         kv positions | rope rows | GDN ctrl page
//   aux_prev_in int32  [7, W_A]         the next fused aux (ovr rows 0..3, p0', ovr row 5, ovr row OVR_DELTA)
// Arithmetic: the op graph computes in fp32 on integers below 2^24, where fp32 is exact, so int32 arithmetic gives the
// same words. A negative value stored to a uint32 output becomes 0, as ttnn.typecast fp32 -> uint32 does (only reachable
// from out-of-contract words, e.g. the all-zero warm-up overlay with rope_rows = 0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t B = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t NV = get_compile_time_arg_val(2);     // GDN value heads per device (ctrl page blocks per user)
constexpr uint32_t W_RES = get_compile_time_arg_val(3);  // words per result_in row
constexpr uint32_t W_A = get_compile_time_arg_val(4);    // words per aux row
constexpr uint32_t W_O = get_compile_time_arg_val(5);    // words per overlay row
constexpr uint32_t W_V = get_compile_time_arg_val(6);    // words per vaux row
constexpr uint32_t DRAFT_PAGE = get_compile_time_arg_val(7);  // bytes read per draft id page (<= 64)
constexpr uint32_t CB_L1 = get_compile_time_arg_val(8);
constexpr uint32_t CB_BYTES = get_compile_time_arg_val(9);
constexpr uint32_t CTA_ACCESSORS = 10;

constexpr uint32_t T = K + 1;
constexpr uint32_t R = B * T;

// spec_fused.AUX_* / CONST_* and spec_chain.RES_* / OVR_* rows
constexpr uint32_t AUX_NUM_VALID = 0, AUX_ACTIVE = 1, AUX_P0 = 4, AUX_CONSTS = 5, AUX_DELTA = 6, AUX_ROWS = 7;
constexpr uint32_t CONST_ROPE_ROWS = 2;
constexpr uint32_t RES_COMMITTED = 0, RES_COUNTS = 1;
constexpr uint32_t OVR_MASK = 6, OVR_TOKENS = 7, OVR_MI = 8, OVR_WORDS = 9, OVR_DELTA = 10, OVR_ROWS = 11;

static_assert(K >= 1 && R <= 32 && B >= 1, "one verify bucket of at most 32 rows");
static_assert(R <= W_RES && B <= W_RES, "result row width");
static_assert(B <= W_A && 5 <= W_A && W_A <= W_O && R <= W_O, "aux / overlay widths");
static_assert(R <= W_V && 1 + B + B * NV <= W_V, "vaux width");
static_assert(DRAFT_PAGE >= 4 && DRAFT_PAGE <= 64, "one draft id per <= 64 B page");

constexpr uint32_t al64(uint32_t x) { return (x + 63u) & ~63u; }

// L1 layout, byte offsets from the 64 B aligned CB base; every row starts on a 64 B line
constexpr uint32_t RES_ROW = al64(W_RES * 4);
constexpr uint32_t AUX_ROW = al64(W_A * 4);
constexpr uint32_t OVR_ROW = al64(W_O * 4);
constexpr uint32_t V_ROW = al64(W_V * 4);
constexpr uint32_t OFF_RES = 0;                            // 2 rows (committed, counts)
constexpr uint32_t OFF_AUXP = OFF_RES + 2 * RES_ROW;       // 1 row (aux_prev row AUX_P0)
constexpr uint32_t OFF_OVR = OFF_AUXP + AUX_ROW;           // OVR_ROWS rows
constexpr uint32_t OFF_DR = OFF_OVR + OVR_ROWS * OVR_ROW;  // K*B slots of 64 B (draft k of user u at slot k*B+u)
constexpr uint32_t OFF_TOK = OFF_DR + K * B * 64;          // 1 row of R words
constexpr uint32_t OFF_VAUX = OFF_TOK + al64(R * 4);       // 3 rows
constexpr uint32_t OFF_NAUX = OFF_VAUX + 3 * V_ROW;        // AUX_ROWS rows
constexpr uint32_t L1_BYTES = OFF_NAUX + AUX_ROWS * AUX_ROW;
static_assert(L1_BYTES + 64 <= CB_BYTES, "scratch CB too small");

using wptr = volatile tt_l1_ptr uint32_t*;
using iptr = volatile tt_l1_ptr int32_t*;

FORCE_INLINE uint32_t u32_sat(int32_t v) { return v < 0 ? 0u : static_cast<uint32_t>(v); }

}  // namespace

void kernel_main() {
    const uint32_t res_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t aux_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ovr_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tok_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t vaux_addr = get_common_arg_val<uint32_t>(4);
    constexpr uint32_t ARG_DRAFT0 = 5;  // K draft addresses follow

    constexpr auto res_args = TensorAccessorArgs<CTA_ACCESSORS>();
    constexpr auto aux_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto ovr_args = TensorAccessorArgs<aux_args.next_compile_time_args_offset()>();
    constexpr auto tok_args = TensorAccessorArgs<ovr_args.next_compile_time_args_offset()>();
    constexpr auto vaux_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto dr_args = TensorAccessorArgs<vaux_args.next_compile_time_args_offset()>();
    const auto res = TensorAccessor(res_args, res_addr);
    const auto aux = TensorAccessor(aux_args, aux_addr);
    const auto ovr = TensorAccessor(ovr_args, ovr_addr);
    const auto tok = TensorAccessor(tok_args, tok_addr);
    const auto vaux = TensorAccessor(vaux_args, vaux_addr);

    const uint32_t base = al64(get_write_ptr(CB_L1));

    // ---- all reads first (aux_prev_in is also an output: its p0 row must be in L1 before any write)
    noc_async_read(res.get_noc_addr(RES_COMMITTED), base + OFF_RES, W_RES * 4);
    noc_async_read(res.get_noc_addr(RES_COUNTS), base + OFF_RES + RES_ROW, W_RES * 4);
    noc_async_read(aux.get_noc_addr(AUX_P0), base + OFF_AUXP, W_A * 4);
    for (uint32_t i = 0; i < OVR_ROWS; ++i) {
        noc_async_read(ovr.get_noc_addr(i), base + OFF_OVR + i * OVR_ROW, W_O * 4);
    }
    for (uint32_t k = 0; k < K; ++k) {
        const auto dr = TensorAccessor(dr_args, get_common_arg_val<uint32_t>(ARG_DRAFT0 + k));
        for (uint32_t u = 0; u < B; ++u) {
            noc_async_read(dr.get_noc_addr(u), base + OFF_DR + (k * B + u) * 64, DRAFT_PAGE);
        }
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    const iptr committed = reinterpret_cast<iptr>(base + OFF_RES);
    const iptr counts = reinterpret_cast<iptr>(base + OFF_RES + RES_ROW);
    const iptr p0_prev = reinterpret_cast<iptr>(base + OFF_AUXP);
    auto orow = [&](uint32_t i) { return reinterpret_cast<iptr>(base + OFF_OVR + i * OVR_ROW); };
    const iptr nv_r = orow(AUX_NUM_VALID);
    const iptr act_r = orow(AUX_ACTIVE);
    const iptr p0h_r = orow(AUX_P0);
    const iptr mask_r = orow(OVR_MASK);
    const iptr htok_r = orow(OVR_TOKENS);
    const iptr mih_r = orow(OVR_MI);
    const iptr delta_r = orow(OVR_DELTA);
    const int32_t rows_m1 = orow(AUX_CONSTS)[CONST_ROPE_ROWS] - 1;
    const int32_t parity = orow(OVR_WORDS)[0];

    const wptr tok_o = reinterpret_cast<wptr>(base + OFF_TOK);
    const wptr pos_o = reinterpret_cast<wptr>(base + OFF_VAUX);
    const wptr ridx_o = reinterpret_cast<wptr>(base + OFF_VAUX + V_ROW);
    const wptr ctrl_o = reinterpret_cast<wptr>(base + OFF_VAUX + 2 * V_ROW);
    const iptr naux = reinterpret_cast<iptr>(base + OFF_NAUX);
    constexpr uint32_t NAUX_STRIDE = AUX_ROW / 4;

    // zero padding of every output row (the pages are written whole)
    for (uint32_t i = 0; i < R; ++i) {
        tok_o[i] = 0;
    }
    for (uint32_t i = 0; i < W_V; ++i) {
        pos_o[i] = 0;
        ridx_o[i] = 0;
        ctrl_o[i] = 0;
    }

    ctrl_o[0] = u32_sat(parity);
    for (uint32_t u = 0; u < B; ++u) {
        const int32_t nv = nv_r[u];
        const int32_t act = act_r[u];
        const int32_t mask = mask_r[u];
        const int32_t nm = 1 - mask;
        const int32_t cnt = counts[u];
        const int32_t m = cnt - 1;
        // p0' = ((p0_prev + counts) * (1 - mask) + p0_host * mask) * active'; mi' likewise from m and mi_host
        const int32_t p0n = ((p0_prev[u] + cnt) * nm + p0h_r[u] * mask) * act;
        const int32_t mi = (m * nm + mih_r[u] * mask) * act;
        // tok0 = committed[u, m] (sum of one_hot(m) * committed: 0 when m is outside [0, T))
        const int32_t tok0 = (m >= 0 && m < static_cast<int32_t>(T)) ? committed[u * T + m] : 0;
        const int32_t delta = delta_r[u];
        for (uint32_t j = 0; j < T; ++j) {
            int32_t tdev = tok0;  // column 0, and draft columns at or past nv'
            if (j > 0 && static_cast<int32_t>(j - 1) < nv) {
                const uint32_t d = *reinterpret_cast<wptr>(base + OFF_DR + ((j - 1) * B + u) * 64);
                tdev = static_cast<int32_t>(d);
            }
            const uint32_t r = u * T + j;
            tok_o[r] = u32_sat((tdev * nm + htok_r[r] * mask) * act);
            const int32_t pos = p0n + static_cast<int32_t>(j);
            pos_o[r] = u32_sat(pos);
            int32_t x = pos + delta;  // min(relu(pos + delta), rope_rows - 1)
            x = x < 0 ? 0 : x;
            x = x < rows_m1 ? x : rows_m1;
            ridx_o[r] = u32_sat(x);
        }
        ctrl_o[1 + u] = u32_sat(mi);
        for (uint32_t h = 0; h < NV; ++h) {
            ctrl_o[1 + B + u * NV + h] = u32_sat((mi * static_cast<int32_t>(B) + static_cast<int32_t>(u)) *
                                                     static_cast<int32_t>(NV) +
                                                 static_cast<int32_t>(h));
        }
        naux[AUX_P0 * NAUX_STRIDE + u] = p0n;
    }
    // next fused aux: overlay rows 0..3 and 5 as they are, row 4 = p0', row 6 = overlay row OVR_DELTA
    for (uint32_t i = 0; i < W_A; ++i) {
        for (uint32_t row = 0; row < AUX_P0; ++row) {
            naux[row * NAUX_STRIDE + i] = orow(row)[i];
        }
        if (i >= B) {
            naux[AUX_P0 * NAUX_STRIDE + i] = 0;
        }
        naux[AUX_CONSTS * NAUX_STRIDE + i] = orow(AUX_CONSTS)[i];
        naux[AUX_DELTA * NAUX_STRIDE + i] = delta_r[i];
    }
    asm volatile("" ::: "memory");

    noc_async_write(base + OFF_TOK, tok.get_noc_addr(0), R * 4);
    for (uint32_t i = 0; i < 3; ++i) {
        noc_async_write(base + OFF_VAUX + i * V_ROW, vaux.get_noc_addr(i), W_V * 4);
    }
    for (uint32_t i = 0; i < AUX_ROWS; ++i) {
        noc_async_write(base + OFF_NAUX + i * AUX_ROW, aux.get_noc_addr(i), W_A * 4);
    }
    noc_async_write_barrier();
}
