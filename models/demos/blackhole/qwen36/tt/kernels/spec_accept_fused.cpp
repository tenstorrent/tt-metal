// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_SPEC_ACCEPT_KERNEL (tt/spec_fused_kernels.py): the S1 greedy accept trace body (spec_fused.FusedGreedyDevice.body,
// 115 ttnn ops of int32 glue and fp32 TILE elementwise math) as ONE data-movement kernel. Same buffers, same words.
//
// Inputs (ROW_MAJOR, one page per row):
//   ids_in     uint32 [1,1,1,32]  verify argmax ids, word r = u*T + j
//   tok_in     uint32 [1, R]      the verify token block (column 0 pending, columns 1..K drafts)
//   aux_in     int32  [7, W]      spec_fused.fused_accept_aux_host layout
//   pt_rows_in int32  [R, NB]     page-table row pt[r // T] per engine row
// Outputs (every word written, zero padding included, as the op graph's ttnn.copy does):
//   pack_out   uint32 [2K+5, W]   drafter input pack (spec_fused.fused_greedy_reference .pack)
//   rs_pt_out  int32  [R, NB]     reseed page table: a real row keeps its pt row, the others are all scratch
//   result_out int32  [2, W]      row 0 committed[R] (-1 past the count), row 1 counts[B]
// Grid: N_CORES cores, logical (x, y) with x < NX, core index c = y * NX + x. Every core reads the small inputs and runs
// the greedy walk (a few hundred integer ops). Core 0 writes pack_out and result_out. The rs_pt rows are cut into
// CHUNK-byte column chunks; core c writes chunks c, c + N_CORES, ... of every row (real rows: copied from pt_rows_in,
// the others: scratch words), so each core fills only one small scratch chunk and moves short pieces.
// Arithmetic: the op graph computes in fp32 on integers below 2^24, where fp32 is exact, so int32 arithmetic gives the
// same words; a negative value stored to a uint32 output becomes 0, as ttnn.typecast fp32 -> uint32 does (reachable only
// from out-of-contract words, e.g. the all-zero warm-up aux with rope_rows = 0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t B = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t W = get_compile_time_arg_val(2);   // words per aux / pack / result row
constexpr uint32_t NB = get_compile_time_arg_val(3);  // words per page-table row
constexpr uint32_t NX = get_compile_time_arg_val(4);  // worker grid width
constexpr uint32_t N_CORES = get_compile_time_arg_val(5);
constexpr uint32_t CHUNK = get_compile_time_arg_val(6);  // rs_pt column chunk, bytes (multiple of 64)
constexpr uint32_t CB_L1 = get_compile_time_arg_val(7);
constexpr uint32_t CB_BYTES = get_compile_time_arg_val(8);
constexpr uint32_t CTA_ACCESSORS = 9;

constexpr uint32_t T = K + 1;
constexpr uint32_t R = B * T;
constexpr uint32_t PACK_ROWS = 2 * K + 5;
constexpr uint32_t ROW_BYTES = NB * 4;
constexpr uint32_t N_CHUNKS = (ROW_BYTES + CHUNK - 1) / CHUNK;

// spec_fused.AUX_* / CONST_* rows
constexpr uint32_t AUX_NUM_VALID = 0, AUX_DECLINED = 2, AUX_KEEP = 3, AUX_P0 = 4, AUX_CONSTS = 5, AUX_DELTA = 6;
constexpr uint32_t AUX_ROWS = 7;
constexpr uint32_t CONST_SCRATCH = 0, CONST_ROPE_ROWS = 2;

static_assert(K >= 1 && B >= 1 && R <= 32, "one 32-lane ids row");
static_assert(R <= W && B <= W && 5 <= W, "row width");
static_assert(CHUNK % 64 == 0 && CHUNK >= 64, "64 B aligned column chunks");
static_assert(N_CORES >= 1 && NX >= 1 && NB >= 1, "grid / table");

constexpr uint32_t al64(uint32_t x) { return (x + 63u) & ~63u; }

// L1 layout, byte offsets from the 64 B aligned CB base
constexpr uint32_t ROW = al64(W * 4);
constexpr uint32_t OFF_IDS = 0;                      // 32 words
constexpr uint32_t OFF_TOK = OFF_IDS + 128;          // R words
constexpr uint32_t OFF_AUX = OFF_TOK + al64(R * 4);  // AUX_ROWS rows
constexpr uint32_t OFF_PACK = OFF_AUX + AUX_ROWS * ROW;
constexpr uint32_t OFF_RES = OFF_PACK + PACK_ROWS * ROW;
constexpr uint32_t OFF_SCR = OFF_RES + 2 * ROW;  // one chunk of scratch words
constexpr uint32_t OFF_PT = OFF_SCR + CHUNK;     // R chunks: the real rows' pt pieces
constexpr uint32_t L1_BYTES = OFF_PT + R * CHUNK;
static_assert(L1_BYTES + 64 <= CB_BYTES, "scratch CB too small");

using wptr = volatile tt_l1_ptr uint32_t*;
using iptr = volatile tt_l1_ptr int32_t*;

FORCE_INLINE uint32_t u32_sat(int32_t v) { return v < 0 ? 0u : static_cast<uint32_t>(v); }

FORCE_INLINE int32_t rope_row(int32_t x, int32_t delta, int32_t rows_m1) {  // min(relu(x + delta), rope_rows - 1)
    int32_t v = x + delta;
    v = v < 0 ? 0 : v;
    return v < rows_m1 ? v : rows_m1;
}

}  // namespace

void kernel_main() {
    const uint32_t core = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    if (core >= N_CORES) {
        return;
    }
    const uint32_t ids_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tok_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t aux_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t pt_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t pack_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t rs_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t res_addr = get_common_arg_val<uint32_t>(6);

    constexpr auto ids_args = TensorAccessorArgs<CTA_ACCESSORS>();
    constexpr auto tok_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto aux_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<aux_args.next_compile_time_args_offset()>();
    constexpr auto pack_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto rs_args = TensorAccessorArgs<pack_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<rs_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto tok = TensorAccessor(tok_args, tok_addr);
    const auto aux = TensorAccessor(aux_args, aux_addr);
    const auto pt = TensorAccessor(pt_args, pt_addr);
    const auto pack = TensorAccessor(pack_args, pack_addr);
    const auto rs = TensorAccessor(rs_args, rs_addr);
    const auto res = TensorAccessor(res_args, res_addr);

    const uint32_t base = al64(get_write_ptr(CB_L1));

    // ---- small inputs (every core)
    noc_async_read(ids.get_noc_addr(0), base + OFF_IDS, 128);
    noc_async_read(tok.get_noc_addr(0), base + OFF_TOK, R * 4);
    for (uint32_t i = 0; i < AUX_ROWS; ++i) {
        noc_async_read(aux.get_noc_addr(i), base + OFF_AUX + i * ROW, W * 4);
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    const wptr ids_w = reinterpret_cast<wptr>(base + OFF_IDS);
    const wptr tok_w = reinterpret_cast<wptr>(base + OFF_TOK);
    auto arow = [&](uint32_t i) { return reinterpret_cast<iptr>(base + OFF_AUX + i * ROW); };
    const iptr nv_r = arow(AUX_NUM_VALID);
    const iptr dec_r = arow(AUX_DECLINED);
    const iptr keep_r = arow(AUX_KEEP);
    const iptr p0_r = arow(AUX_P0);
    const iptr delta_r = arow(AUX_DELTA);
    const int32_t scratch = arow(AUX_CONSTS)[CONST_SCRATCH];
    const int32_t rows_m1 = arow(AUX_CONSTS)[CONST_ROPE_ROWS] - 1;

    // ---- greedy walk: m = length of the valid draft prefix that matches the argmax ids (sum of cumprod(acc));
    // the real (reseeded) rows of user u are j < m + declined * keep
    uint32_t m_of[B];
    uint32_t real_of[B];
    for (uint32_t u = 0; u < B; ++u) {
        const int32_t nv = nv_r[u];
        uint32_t m = 0;
        while (m < K && static_cast<int32_t>(m) < nv && ids_w[u * T + m] == tok_w[u * T + m + 1]) {
            ++m;
        }
        m_of[u] = m;
        const int32_t kv = static_cast<int32_t>(m) + dec_r[u] * keep_r[u];
        real_of[u] = kv <= 0 ? 0u : (kv >= static_cast<int32_t>(T) ? T : static_cast<uint32_t>(kv));
    }

    // ---- rs_pt: this core's column chunks of every row
    if (core < N_CHUNKS) {
        const wptr scr = reinterpret_cast<wptr>(base + OFF_SCR);
        bool filled = false;
        for (uint32_t ch = core; ch < N_CHUNKS; ch += N_CORES) {
            const uint32_t lo = ch * CHUNK;
            const uint32_t n = lo + CHUNK <= ROW_BYTES ? CHUNK : ROW_BYTES - lo;
            if (ch != core) {
                noc_async_writes_flushed();  // the previous chunk's pt pieces left L1
            }
            for (uint32_t r = 0; r < R; ++r) {
                if (r % T < real_of[r / T]) {
                    noc_async_read(pt.get_noc_addr(r, lo), base + OFF_PT + r * CHUNK, n);
                }
            }
            if (!filled) {  // while the reads fly
                for (uint32_t i = 0; i < CHUNK / 4; ++i) {
                    scr[i] = static_cast<uint32_t>(scratch);
                }
                asm volatile("" ::: "memory");
                filled = true;
            }
            noc_async_read_barrier();
            for (uint32_t r = 0; r < R; ++r) {
                const bool real = r % T < real_of[r / T];
                noc_async_write(base + (real ? OFF_PT + r * CHUNK : OFF_SCR), rs.get_noc_addr(r, lo), n);
            }
        }
    }

    if (core == 0) {
        // ---- pack [2K+5, W] and result [2, W]
        const wptr pk = reinterpret_cast<wptr>(base + OFF_PACK);
        const iptr rr = reinterpret_cast<iptr>(base + OFF_RES);
        constexpr uint32_t S = ROW / 4;  // words per L1 row
        for (uint32_t i = 0; i < PACK_ROWS * S; ++i) {
            pk[i] = 0;
        }
        for (uint32_t i = 0; i < 2 * S; ++i) {
            rr[i] = 0;
        }
        constexpr uint32_t G1 = 0;      // rows 0..K: pending, then steps k = p + k
        constexpr uint32_t G2 = K + 1;  // rtok, rpos
        constexpr uint32_t G3 = K + 3;  // anchor row u*T + m
        constexpr uint32_t G4 = K + 4;  // ridx(rpos)
        constexpr uint32_t G5 = K + 5;  // rows ridx(steps k)
        for (uint32_t u = 0; u < B; ++u) {
            const uint32_t m = m_of[u];
            const int32_t ndec = 1 - dec_r[u];
            const int32_t p0 = p0_r[u];
            const int32_t delta = delta_r[u];
            const int32_t p = (p0 + static_cast<int32_t>(m)) * ndec;
            pk[G1 * S + u] = u32_sat(static_cast<int32_t>(ids_w[u * T + m]) * ndec);
            for (uint32_t k = 0; k < K; ++k) {
                const int32_t step = p + static_cast<int32_t>(k);
                pk[(G1 + 1 + k) * S + u] = u32_sat(step);
                pk[(G5 + k) * S + u] = u32_sat(rope_row(step, delta, rows_m1));
            }
            pk[G3 * S + u] = u * T + m;
            for (uint32_t j = 0; j < T; ++j) {
                const uint32_t r = u * T + j;
                const int32_t cm = j <= m ? static_cast<int32_t>(ids_w[r]) : -1;
                const int32_t real = j < real_of[u] ? 1 : 0;
                const int32_t rpos = (p0 + static_cast<int32_t>(j)) * real;
                pk[G2 * S + r] = u32_sat(cm * real);
                pk[(G2 + 1) * S + r] = u32_sat(rpos);
                pk[G4 * S + r] = u32_sat(rope_row(rpos, delta, rows_m1));
                rr[r] = cm;
            }
            rr[S + u] = static_cast<int32_t>(m + 1);
        }
        asm volatile("" ::: "memory");
        for (uint32_t i = 0; i < PACK_ROWS; ++i) {
            noc_async_write(base + OFF_PACK + i * ROW, pack.get_noc_addr(i), W * 4);
        }
        noc_async_write(base + OFF_RES, res.get_noc_addr(0), W * 4);
        noc_async_write(base + OFF_RES + ROW, res.get_noc_addr(1), W * 4);
    }
    noc_async_write_barrier();
}
