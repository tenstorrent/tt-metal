// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused block-scaled quantize-dequantize of bf16 activations (DeepSeek-V4.1, tt/v41/qdq.py).
//
// One quantization group is 32 (one tile row) or 16 (one face row: tile columns 0-15 or 16-31) consecutive
// elements of a row. Per block of up to BLOCK tiles:
//   A  |x| -> cb_abs.
//   B  row max of |x| (FPU reduce; for group 16 twice, with scaler tiles that mask the other half's columns)
//      -> group amax in column 0 -> scale (SFPU, per format) -> cb_scale.
//   C  column-broadcast of the scale (for group 16 the two halves are merged by face) -> cb_scale_bcast.
//   D  x and its broadcast scale -> quantize-dequantize (SFPU, per format) -> cb_out.
// All CBs are bf16. Every value that leaves DST is exact in bf16 (|x|, amax, power-of-two or e4m3 scales, and
// dequantized values with <= 6 significant bits), so no rounding happens outside the SFPU math below.
// fp32_dest_acc_en = true: the SFPU math runs on fp32 copies of the bf16 inputs.
//
// compile_time_args = [QDQ_FORMAT, BLOCK]; runtime args = [num_tiles].
//   QDQ_FORMAT 0: FP8 e4m3, group 32, power-of-two (ue8m0) scale      (act_quant(x, 32, "ue8m0"))
//   QDQ_FORMAT 1: FP4 e2m1, group 32, power-of-two (E8M0) scale       (fp4_act_quant(x, 32))
//   QDQ_FORMAT 2: FP4 e2m1, group 16, e4m3 scale                      (fp4_act_quant(x, 16, e4m3 scale))

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// Exactness notes (the reasoning behind tt/v41/qdq.py's docstring, restated for this code):
// * SFPU fp32 add/mul/compare are exact whenever the exact result is representable; every product or sum
//   formed below has at most 8 significant bits and a normal exponent, except where noted.
// * e4m3 rounding: values below 2^-6 (the e4m3 subnormal range, quantum 2^-9) are offset by 2^-6, which moves
//   them onto the normal binade [2^-6, 2^-5) with the same quantum 2^-9 and the same code parity, so one
//   "clear the low 20 mantissa bits" floor serves both ranges. The rounding decision compares the exact value
//   against the half-way point instead of trusting the (possibly inexact) offset estimate.

constexpr int QDQ_EXP_MASK = 0x7F800000;
constexpr int QDQ_MANT_MASK = 0x007FFFFF;
constexpr int QDQ_E4M3_FLOOR_MASK = static_cast<int>(0xFFF00000u);  // keeps 3 mantissa bits
constexpr float QDQ_E4M3_MIN_NORMAL = 0.015625f;                    // 2^-6
constexpr float QDQ_E4M3_MAX = 448.0f;

// Nearest e4m3 magnitude (RNE, no saturation) of the exact non-negative value v, where
// v == ref (TIMES6 = false) or v == ref / 6 (TIMES6 = true), and est is within a few ulp of v.
template <bool TIMES6>
sfpi_inline sfpi::vFloat qdq_round_e4m3(sfpi::vFloat est, sfpi::vFloat ref) {
    sfpi::vFloat off = 0.0f;
    v_if(est < QDQ_E4M3_MIN_NORMAL) { off = QDQ_E4M3_MIN_NORMAL; }
    v_endif;
    sfpi::vInt w = sfpi::as<sfpi::vInt>(est + off);
    sfpi::vInt lower_bits = w & QDQ_E4M3_FLOOR_MASK;
    sfpi::vFloat lower = sfpi::as<sfpi::vFloat>(lower_bits);
    sfpi::vFloat half_quantum = sfpi::as<sfpi::vFloat>((w & QDQ_EXP_MASK) - (4 << 23));
    sfpi::vFloat midpoint = (lower + half_quantum) - off;  // exact: <= 5 significant bits
    if constexpr (TIMES6) {
        midpoint = midpoint * 6.0f;  // exact: <= 7 significant bits
    }
    sfpi::vFloat result = lower - off;
    sfpi::vFloat upper = (lower + (half_quantum + half_quantum)) - off;
    v_if(ref > midpoint) { result = upper; }
    v_elseif(ref == midpoint) {
        // exact tie: round to the even code (bit 20 is the lowest kept mantissa bit)
        v_if((lower_bits & (1 << 20)) != 0) { result = upper; }
        v_endif;
    }
    v_endif;
    return result;
}

// RNE of an exact non-negative v with at most 8 significant bits onto a float grid that keeps 23 - DROP mantissa
// bits and whose smallest normal is MIN_NORMAL (e4m3: DROP 20, 2^-6; E2M1: DROP 22, 1): below MIN_NORMAL the grid
// quantum stays that of [MIN_NORMAL, 2 MIN_NORMAL), so v is offset onto that binade (same quantum, same code parity)
// and rounded there with the integer RNE of its bit pattern. v + MIN_NORMAL is exact unless v < MIN_NORMAL * 2^-16,
// where the (inexact) sum is still within 2^-16 of MIN_NORMAL and rounds to it, i.e. v rounds to 0 as it must.
template <int DROP, float MIN_NORMAL>
sfpi_inline sfpi::vFloat qdq_round_exact(sfpi::vFloat v) {
    sfpi::vFloat off = 0.0f;
    v_if(v < MIN_NORMAL) { off = MIN_NORMAL; }
    v_endif;
    sfpi::vInt w = sfpi::as<sfpi::vInt>(v + off);
    w = w + (((1 << (DROP - 1)) - 1)) + ((w >> DROP) & 1);
    w = w & static_cast<int>(~((1u << DROP) - 1));
    return sfpi::as<sfpi::vFloat>(w) - off;
}

// fp32 2^(exponent(amax) - BIAS + [mantissa(amax) > THRESHOLD]), floored at FLOOR_BITS (fp32 bits of a power of
// two). Equals fast_round_scale(amax / M) (M = 448: BIAS 8, THRESHOLD 1.75; M = 6: BIAS 2, THRESHOLD 1.5) for bf16
// amax, then the reference's amax floors (see qdq.py).
template <int BIAS, int THRESHOLD, int FLOOR_BITS>
sfpi_inline sfpi::vFloat qdq_pow2_scale(sfpi::vFloat amax) {
    sfpi::vInt bits = sfpi::as<sfpi::vInt>(amax);
    sfpi::vInt s = (bits & QDQ_EXP_MASK) - (BIAS << 23);
    v_if((bits & QDQ_MANT_MASK) > THRESHOLD) { s = s + (1 << 23); }
    v_endif;
    v_if(s < FLOOR_BITS) { s = FLOOR_BITS; }
    v_endif;
    return sfpi::as<sfpi::vFloat>(s);
}

// The scale travels DST -> bf16 CB -> broadcast (SrcB) -> DST. That path is not exact for tiny values (measured:
// a bf16 2^-119 comes back about 2^-127 too large), and the FP4 power-of-two scale goes down to 2^-126. So the scale
// travels as the float whose bits are (bits(s) >> 1) + 64 << 23, in [2^-63, 2^64], and is decoded after the
// broadcast; the <= 4 significant bits of s (exponent parity + 3 mantissa bits) stay within bf16's 7.
sfpi_inline sfpi::vFloat qdq_encode_scale(sfpi::vFloat s) {
    return sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vInt>(s) >> 1) + (64 << 23));
}

sfpi_inline sfpi::vFloat qdq_decode_scale(sfpi::vFloat u) {
    return sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vInt>(u) - (64 << 23)) << 1);
}

// Group amax (fp32, in place) -> the format's scale, encoded (qdq_encode_scale).
template <int FORMAT, int ITERATIONS = 8>
inline void calculate_qdq_scale() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat amax = sfpi::dst_reg[0];
        sfpi::vFloat s;
        if constexpr (FORMAT == 0) {
            s = qdq_pow2_scale<8, 0x600000, (-22 + 127) << 23>(amax);
        } else if constexpr (FORMAT == 1) {
            s = qdq_pow2_scale<2, 0x400000, (-126 + 127) << 23>(amax);
        } else {
            // e4m3_satfinite(max(amax, 6 * 2^-9) / 6), from the estimate amax * fp32(1/6)
            v_if(amax < 0.01171875f) { amax = 0.01171875f; }
            v_endif;
            sfpi::vFloat sixth = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x3E2AAAAB));
            s = qdq_round_e4m3<true>(amax * sixth, amax);
            v_if(s > QDQ_E4M3_MAX) { s = QDQ_E4M3_MAX; }
            v_endif;
        }
        sfpi::dst_reg[0] = qdq_encode_scale(s);
        sfpi::dst_reg++;
    }
}

// Nearest E2M1 magnitude (RNE) of min(v, 6) for v >= 0, where v > c is evaluated as `above(c)` / `at_or_above(c)`.
#define QDQ_E2M1(GT, GE)          \
    sfpi::vFloat q = 0.0f;        \
    v_if(GT(0.25f)) { q = 0.5f; } \
    v_endif;                      \
    v_if(GE(0.75f)) { q = 1.0f; } \
    v_endif;                      \
    v_if(GT(1.25f)) { q = 1.5f; } \
    v_endif;                      \
    v_if(GE(1.75f)) { q = 2.0f; } \
    v_endif;                      \
    v_if(GT(2.5f)) { q = 3.0f; }  \
    v_endif;                      \
    v_if(GE(3.5f)) { q = 4.0f; }  \
    v_endif;                      \
    v_if(GT(5.0f)) { q = 6.0f; }  \
    v_endif;

// x (dst x_idx) and its group's scale broadcast over the group (dst s_idx) -> sign(x) * q * s in dst out_idx.
template <int FORMAT, int ITERATIONS = 8>
inline void calculate_qdq(const uint x_idx, const uint s_idx, const uint out_idx) {
    constexpr uint dst_tile_size = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[x_idx * dst_tile_size];
        sfpi::vFloat s = qdq_decode_scale(sfpi::dst_reg[s_idx * dst_tile_size]);
        sfpi::vFloat m = sfpi::abs(x);
        sfpi::vFloat r;
        if constexpr (FORMAT == 2) {
            // non-power-of-two scale: compare |x| against boundary * s (exact products) instead of forming |x| / s
#define QDQ_GT(c) m > s*(c)
#define QDQ_GE(c) m >= s*(c)
            QDQ_E2M1(QDQ_GT, QDQ_GE)
#undef QDQ_GT
#undef QDQ_GE
            r = q * s;
        } else {
            // |x| / s == |x| * (1 / s) exactly for a power-of-two s (1/s from the exponent bits)
            sfpi::vInt inv_bits = sfpi::vInt(254 << 23) - sfpi::as<sfpi::vInt>(s);
            sfpi::vFloat v = m * sfpi::as<sfpi::vFloat>(inv_bits);
            // v <= 448 (FP8) / 6 (FP4) by construction of the scale, so no saturation is needed
            if constexpr (FORMAT == 0) {
                r = qdq_round_exact<20, QDQ_E4M3_MIN_NORMAL>(v) * s;
            } else {
                r = qdq_round_exact<22, 1.0f>(v) * s;
            }
        }
        sfpi::dst_reg[out_idx * dst_tile_size] = sfpi::copysgn(r, x);
        sfpi::dst_reg++;
    }
}
#undef QDQ_E2M1

// Group 16: the left-half scale (tile columns 0-15: faces 0, 2) is in dst l_idx, the right-half one (faces 1, 3) in
// dst r_idx, each broadcast over all columns; copy faces 1 and 3 of r_idx into out_idx (== l_idx). Called once per
// tile (VectorMode::None); a face is 8 SFPU rows.
inline void calculate_qdq_merge_halves(const uint l_idx, const uint r_idx, const uint out_idx) {
    constexpr uint dst_tile_size = 32;
#pragma GCC unroll 2
    for (uint face = 1; face < 4; face += 2) {
#pragma GCC unroll 8
        for (uint d = 0; d < 8; d++) {
            sfpi::vFloat v = sfpi::dst_reg[r_idx * dst_tile_size + face * 8 + d];
            sfpi::dst_reg[out_idx * dst_tile_size + face * 8 + d] = v;
        }
    }
}

}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t FORMAT = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);  // tiles per block, <= 4
    constexpr uint32_t HALVES = FORMAT == 2 ? 2 : 1;         // groups per tile row
    constexpr uint32_t DST_SLOTS = 4;                        // fp32 half-sync DST
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;  // HALVES reduce scaler tiles (column masks for group 16)
    constexpr uint32_t cb_abs = tt::CBIndex::c_2;
    constexpr uint32_t cb_scale = tt::CBIndex::c_3;
    constexpr uint32_t cb_scale_bcast = tt::CBIndex::c_4;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer in(cb_in);
    CircularBuffer scaler(cb_scaler);
    CircularBuffer abs_cb(cb_abs);
    CircularBuffer scale(cb_scale);
    CircularBuffer scale_bcast(cb_scale_bcast);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_abs, cb_scaler, cb_out);
    scaler.wait_front(HALVES);

    for (uint32_t done = 0; done < num_tiles;) {
        const uint32_t n = num_tiles - done < BLOCK ? num_tiles - done : BLOCK;
        in.wait_front(n);

        // ---- A: |x| ----
        copy_init(cb_in);
        abs_tile_init();
        abs_cb.reserve_back(n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(cb_in, i, i);
            abs_tile(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_abs);
        }
        tile_regs_release();
        abs_cb.push_back(n);

        // ---- B: group amax -> scale ----
        abs_cb.wait_front(n);
        const uint32_t groups = n * HALVES;
        scale.reserve_back(groups);
        for (uint32_t g0 = 0; g0 < groups; g0 += DST_SLOTS) {
            const uint32_t k = groups - g0 < DST_SLOTS ? groups - g0 : DST_SLOTS;
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_scaler, cb_scale);
            tile_regs_acquire();
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t g = g0 + j;
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_scaler, g / HALVES, g % HALVES, j);
            }
            reduce_uninit();
            MATH((SFPU_UNARY_INIT(unused)));
            for (uint32_t j = 0; j < k; ++j) {
                // column 0 (the reduce output) lives in faces 0 and 2
                MATH((SFPU_UNARY_CALL(
                    DST_SYNC_MODE, DST_ACCUM_MODE, calculate_qdq_scale, (FORMAT, 8), j, VectorMode::C)));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < k; ++j) {
                pack_tile(j, cb_scale);
            }
            tile_regs_release();
        }
        scale.push_back(groups);
        abs_cb.pop_front(n);

        // ---- C: broadcast the scale over its group ----
        // (binary SFPU calls address DST slots by constant offsets, so their slots are fixed)
        scale.wait_front(groups);
        scale_bcast.reserve_back(n);
        if constexpr (HALVES == 1) {
            unary_bcast_init<BroadcastType::COL>(cb_scale);
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                unary_bcast<BroadcastType::COL>(cb_scale, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_scale_bcast);
            }
            tile_regs_release();
            unary_bcast_uninit<BroadcastType::COL>(cb_scale);
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                unary_bcast_init<BroadcastType::COL>(cb_scale);
                tile_regs_acquire();
                unary_bcast<BroadcastType::COL>(cb_scale, 2 * i, 0);
                unary_bcast<BroadcastType::COL>(cb_scale, 2 * i + 1, 1);
                MATH((SFPU_BINARY_INIT(unused)));
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, calculate_qdq_merge_halves, 0, 1, 0, VectorMode::None)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_scale_bcast);
                tile_regs_release();
                unary_bcast_uninit<BroadcastType::COL>(cb_scale);
            }
        }
        scale_bcast.push_back(n);
        scale.pop_front(groups);

        // ---- D: quantize-dequantize ----
        scale_bcast.wait_front(n);
        out.reserve_back(n);
        copy_init(cb_in);
        MATH((SFPU_BINARY_INIT(unused)));
        for (uint32_t i = 0; i < n; ++i) {
            tile_regs_acquire();
            copy_tile(cb_in, i, 0);
            copy_tile(cb_scale_bcast, i, 1);
            MATH(
                (SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_qdq, (FORMAT, 8), 0, 1, 0, VectorMode::RC)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        out.push_back(n);
        scale_bcast.pop_front(n);
        in.pop_front(n);
        done += n;
    }
}
