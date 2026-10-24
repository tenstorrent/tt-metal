// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// RouterTopk DENSE mode (prefill rows): one data-movement kernel on min(grid, S) cores. Core c handles token rows
// c, c + n_cores, ... of the [S, E] logits (bf16 TILE): per row it reads the row tile's COLS_T logit tiles, ranks
// its row (top-k of the bf16 logits as floats), takes the k exponentials against the row max (fp32 scalar code),
// normalizes by their sum, scales by the per-expert scale and writes the dense routing row (E bf16: the k weights
// at their expert columns, zeros elsewhere) straight into the TILE [S, E] output: per column tile two 32-byte
// face-row segments. Same arithmetic as the decode op (softmax then top-k == top-k of the logits then the k
// exponentials), so the result equals the composite softmax / topk / sum / div / scatter / mul chain.
//
// compile-time args: [cb_logits, cb_scratch, cols_t, E, K, pes_page, n_cores] + TensorAccessorArgs(logits, pes, out)
// runtime args: [logits_addr, pes_addr, out_addr, my_core, S]
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_LOGITS = get_compile_time_arg_val(0);
constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(1);
constexpr uint32_t COLS_T = get_compile_time_arg_val(2);
constexpr uint32_t E = get_compile_time_arg_val(3);
constexpr uint32_t K = get_compile_time_arg_val(4);
constexpr uint32_t PES_PAGE = get_compile_time_arg_val(5);
constexpr uint32_t N_CORES = get_compile_time_arg_val(6);
constexpr auto logits_args = TensorAccessorArgs<7>();
constexpr auto pes_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
constexpr auto out_args = TensorAccessorArgs<pes_args.next_compile_time_args_offset()>();
constexpr uint32_t TILE_BYTES = 2048;
// scratch layout (bytes): pes row | the dense output row (E bf16, face-row segments of 16 elements)
constexpr uint32_t OFF_PES = 0;
constexpr uint32_t OFF_ROW = 512;

union FU32 {
    float f;
    uint32_t u;
};
FORCE_INLINE float bf16_to_f32(uint16_t b) {
    FU32 v;
    v.u = uint32_t(b) << 16;
    return v.f;
}
FORCE_INLINE uint16_t f32_to_bf16(float f) {  // round to nearest even
    FU32 v;
    v.f = f;
    const uint32_t r = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return uint16_t(r >> 16);
}
// monotone unsigned key of a float (totally ordered, -0 < +0)
FORCE_INLINE uint32_t fkey(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
// exp(x) for x <= 0: 2^n * 2^r with r = x*log2e - n in [0, 1), 2^r by a degree-6 polynomial (rel. err < 2e-7)
FORCE_INLINE float exp_neg(float x) {
    if (x < -87.0f) {
        return 0.0f;
    }
    const float y = x * 1.44269504f;
    int n = int(y);
    if (float(n) > y) {
        --n;  // floor for negative y
    }
    const float r = (y - float(n)) * 0.69314718f;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    FU32 s;
    s.u = uint32_t(n + 127) << 23;
    return p * s.f;
}
// element offset of (row r, column j) inside the row tile's tile j/32: face (r/16)*2 + (j%32)/16
FORCE_INLINE uint32_t tile_elem(uint32_t r, uint32_t j) {
    const uint32_t cj = j & 31;
    return ((r >> 4) * 2 + (cj >> 4)) * 256 + (r & 15) * 16 + (cj & 15);
}

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t pes_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t me = get_arg_val<uint32_t>(3);
    const uint32_t S = get_arg_val<uint32_t>(4);
    const uint32_t scratch = get_write_ptr(CB_SCRATCH);
    const uint32_t logits_l1 = get_write_ptr(CB_LOGITS);
    const auto s_logits = TensorAccessor(logits_args, logits_addr, TILE_BYTES);
    const auto s_pes = TensorAccessor(pes_args, pes_addr, PES_PAGE);
    const auto s_out = TensorAccessor(out_args, out_addr, TILE_BYTES);
    noc_async_read(s_pes.get_noc_addr(0), scratch + OFF_PES, PES_PAGE);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* lg = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(logits_l1);
    volatile tt_l1_ptr uint16_t* pes = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_PES);
    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_ROW);
    uint32_t loaded_rt = 0xFFFFFFFFu;
    for (uint32_t tok = me; tok < S; tok += N_CORES) {
        const uint32_t rt = tok >> 5, r = tok & 31;
        if (rt != loaded_rt) {  // the row tile's logit tiles (reused when this core's next row shares them)
            for (uint32_t c = 0; c < COLS_T; ++c) {
                noc_async_read(s_logits.get_noc_addr(rt * COLS_T + c), logits_l1 + c * TILE_BYTES, TILE_BYTES);
            }
            noc_async_read_barrier();
            loaded_rt = rt;
        }
        uint32_t key[E];
        for (uint32_t j = 0; j < E; ++j) {
            key[j] = fkey(uint32_t(lg[(j >> 5) * 1024 + tile_elem(r, j)]) << 16);
        }
        uint32_t sel[K];
        for (uint32_t s = 0; s < K; ++s) {
            uint32_t best = 0xFFFFFFFFu, bkey = 0;
            for (uint32_t j = 0; j < E; ++j) {
                bool taken = false;
                for (uint32_t q = 0; q < s; ++q) {
                    taken |= (sel[q] == j);
                }
                if (!taken && (best == 0xFFFFFFFFu || key[j] > bkey)) {
                    best = j;
                    bkey = key[j];
                }
            }
            sel[s] = best;
        }
        float p[K];
        float sum = 0.0f;
        const float m = bf16_to_f32(lg[(sel[0] >> 5) * 1024 + tile_elem(r, sel[0])]);
        for (uint32_t s = 0; s < K; ++s) {
            const float l = bf16_to_f32(lg[(sel[s] >> 5) * 1024 + tile_elem(r, sel[s])]);
            p[s] = exp_neg(l - m);
            sum += p[s];
        }
        // the dense row, laid out as the 2 * COLS_T face-row segments of 16 elements the writes take
        for (uint32_t j = 0; j < E; ++j) {
            row[j] = 0;
        }
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t g = sel[s];
            row[g] = f32_to_bf16(p[s] / sum * bf16_to_f32(pes[g]));
        }
        for (uint32_t t = 0; t < COLS_T; ++t) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t elem = ((r >> 4) * 2 + half) * 256 + (r & 15) * 16;  // first element of the segment
                noc_async_write(
                    scratch + OFF_ROW + (t * 32 + half * 16) * 2, s_out.get_noc_addr(rt * COLS_T + t) + elem * 2, 32);
            }
        }
        noc_async_write_barrier();
    }
}
