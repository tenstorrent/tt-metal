// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// DST-resident coefficient store (opt-in, gelu-focused pilot).
//
// Multi-segment polynomial cascades pay 2x SFPLOADI per fp32 coefficient
// materialization (per element pair in the dual-eval body). Parking the
// coefficients ONCE PER TILE in unused DST rows (tile 1 of the current
// half-sync half, dst_reg[32..]) turns each materialization into a single
// SFPLOAD with a row-offset immediate.
//
// Residency contract: rows are (re)written at the top of EVERY tile
// iteration, after copy_tile has placed the data tile at dst index 0. This
// sidesteps the pack half-sync double buffer entirely — the parked rows are
// always in the same half the math is currently addressing, so there is no
// stale-other-half hazard at any tile count (the {145,146} ladder points).
// The per-tile cost is ~3 Tensix instructions per parked value, amortized
// over the 16 dual-eval pairs of the tile.
//
// Format contract (bf16 dest mode): coefficients are stored and reloaded
// with the FP32 load/store format (SFP*_MOD0_FMT_FP32), NOT the dest-default
// SRCB format (which truncates fp32->bf16 and would break byte-identity).
// DST_COEFF_PROBE below is the silicon proof for this roundtrip.
//
// This file is included inside `namespace sfpi` by piecewise_generic.cpp.

#pragma once

#if defined(DST_COEFF_PROBE) || defined(DST_COEFF_STORE)

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// First dst_reg index used for parked values: tile 1 of the current half
// (data tile is tile 0 = dst_reg[0..31]).
//
// FUSE_GRAD_MUL moves this to tile 2. It parks the incoming `grad` in DST tile
// 1 (rows 32..63), which is exactly where
// parked coefficients and the asymptotic constants read by asym_fetch<>() used
// to live -- grad clobbered them, asym_fetch returned garbage, and the
// exp(-x^2) path emitted inf/NaN. Measured on erf_bw p1_s1 as
// "32 non-finite hardware outputs have finite references". Shifting the parked
// region to tile 2 keeps both tenants disjoint; the capacity gate below
// (kDstCoeffBase + kTotalParked <= 128) tightens accordingly, so a few
// high-segment configs fall back to the non-DST path rather than corrupt.
#if defined(FUSE_GRAD_MUL)
constexpr int kDstCoeffBase = 64;
#else
constexpr int kDstCoeffBase = 32;
#endif

// Store a value's raw fp32 bits into a DST row (both 16-bit halves).
inline void dst_park_f32(int row, vFloat v) { dst_reg[row].mode<DataLayout::F32>() = v; }

// Reload raw fp32 bits from a DST row.
inline vFloat dst_load_f32(int row) { return dst_reg[row].mode<DataLayout::F32>(); }

#endif  // DST_COEFF_PROBE || DST_COEFF_STORE

#ifdef DST_COEFF_PROBE

// ---------------------------------------------------------------------------
// Silicon probe: does an fp32 SFPSTORE->SFPLOAD roundtrip through a DST row
// survive bit-exactly under bf16 (16-bit) dest mode, and does the FP32-format
// store clobber any neighboring rows?
//
// Output tile layout (each output row is 32 identical lanes, all values are
// small integers or bf16-exact so they survive the bf16 pack):
//   r0      : K1 seen through the normal bf16 output path (sanity)
//   r1-r4   : bytes 3..0 of K1, no roundtrip (decompose machinery control)
//   r5-r8   : bytes 3..0 of K1 after SRCB (dest-default) roundtrip @row 36
//   r9-r12  : bytes 3..0 of K1 after FP32-format roundtrip        @row 40
//   r13-r16 : bytes 3..0 of K1 bits after INT32(vInt) roundtrip   @row 44
//   r17-r20 : bytes 3..0 of K2 after FP32-format roundtrip        @row 48
//   r21     : SRCB (bf16-view) read of row 40 after the FP32 store
//   r22     : SRCB (bf16-view) read of row 41 after the FP32 store
//   r23-r28 : sentinel readback of rows {34, 38, 39, 104, 168, 296}
//             (pre-written as 200+k via SRCB before the FP32 stores)
//   r29     : 0/1 flag — K2 FP32 roundtrip bits != original
//   r30     : 0/1 flag — K1 FP32 roundtrip bits != original
//   r31     : 0/1 flag — K1 INT32 roundtrip bits != original
// ---------------------------------------------------------------------------

inline vFloat probe_byte(vInt bits, int shift) {
    vInt b = as<vInt>(as<vUInt>(bits) >> shift) & 0xFF;
    vFloat y = convert<vFloat>(as<vSMag>(b), RoundMode::Nearest);
    return convert<vFloat16b>(y, RoundMode::Nearest);
}

// Emit one byte of `bits` (already shifted) as an output row, immediately —
// keeps SFPU register liveness minimal (the first probe draft ICE'd GCC's
// register allocator with "cannot store sfpu register (register spill)").
inline void probe_emit_bytes(int row, vInt bits) {
    dst_reg[row + 0] = probe_byte(bits, 24);
    dst_reg[row + 1] = probe_byte(bits, 16);
    dst_reg[row + 2] = probe_byte(bits, 8);
    dst_reg[row + 3] = probe_byte(bits, 0);
}

inline void probe_emit_bf16(int out_row, int src_row) {
    vFloat v = dst_reg[src_row];
    dst_reg[out_row] = convert<vFloat16b>(v, RoundMode::Nearest);
}

inline void dst_coeff_probe() {
    // Two non-bf16-exact constants with distinctive byte patterns.
    constexpr uint32_t kK1 = 0x40490FDBu;  // pi
    constexpr uint32_t kK2 = 0x3EAAAAABu;  // 1/3

    // Sentinels first (SRCB/bf16 stores, exact small ints).
    dst_reg[34] = vFloat(201.0f);
    dst_reg[38] = vFloat(202.0f);
    dst_reg[39] = vFloat(203.0f);
    dst_reg[104] = vFloat(204.0f);
    dst_reg[168] = vFloat(205.0f);
    dst_reg[296] = vFloat(206.0f);

    // Stores under test (K1 into rows 36/40/44 by format, K2 into 48).
    {
        vFloat k1 = ckernel::sfpu::Converter::as_float(kK1);
        dst_reg[36] = k1;                                    // SRCB (dest-default) store
        dst_park_f32(40, k1);                                // FP32-format store
        dst_reg[44].mode<DataLayout::I32>() = as<vInt>(k1);  // INT32
    }
    dst_park_f32(48, ckernel::sfpu::Converter::as_float(kK2));

    // r0: K1 through the normal bf16 output path; r1-4: control bytes.
    {
        vFloat k1 = ckernel::sfpu::Converter::as_float(kK1);
        dst_reg[0] = convert<vFloat16b>(k1, RoundMode::Nearest);
        probe_emit_bytes(1, as<vInt>(k1));
    }
    // r5-8: SRCB roundtrip bytes.
    {
        vFloat rt = dst_reg[36];
        probe_emit_bytes(5, as<vInt>(rt));
    }
    // r9-12: FP32 roundtrip bytes.
    {
        vFloat rt = dst_load_f32(40);
        probe_emit_bytes(9, as<vInt>(rt));
    }
    // r13-16: INT32 roundtrip bytes.
    {
        vInt rt = dst_reg[44].mode<DataLayout::I32>();
        probe_emit_bytes(13, rt);
    }
    // r17-20: K2 FP32 roundtrip bytes.
    {
        vFloat rt = dst_load_f32(48);
        probe_emit_bytes(17, as<vInt>(rt));
    }
    // r21-22: bf16-lens view of rows 40/41 after the FP32 store.
    probe_emit_bf16(21, 40);
    probe_emit_bf16(22, 41);
    // r23-28: sentinel readbacks.
    probe_emit_bf16(23, 34);
    probe_emit_bf16(24, 38);
    probe_emit_bf16(25, 39);
    probe_emit_bf16(26, 104);
    probe_emit_bf16(27, 168);
    probe_emit_bf16(28, 296);
    // r29: K2 FP32 flag; r30: K1 FP32 flag; r31: K1 INT32 flag.
    {
        vInt d = as<vInt>(dst_load_f32(48)) ^ (vInt)kK2;
        vFloat f = 0.0f;
        v_if(d != 0) { f = 1.0f; }
        v_endif;
        dst_reg[29] = f;
    }
    {
        vInt d = as<vInt>(dst_load_f32(40)) ^ (vInt)kK1;
        vFloat f = 0.0f;
        v_if(d != 0) { f = 1.0f; }
        v_endif;
        dst_reg[30] = f;
    }
    {
        vInt rt = dst_reg[44].mode<DataLayout::I32>();
        vInt d = rt ^ (vInt)kK1;
        vFloat f = 0.0f;
        v_if(d != 0) { f = 1.0f; }
        v_endif;
        dst_reg[31] = f;
    }
}

#endif  // DST_COEFF_PROBE
// ===========================================================================
// The real DST-resident coefficient store body (dual-eval dense cascade).
// ===========================================================================
//
// Shape gate: the dense (non-parity) dual-eval polynomial cascade with an
// optional EXP_QUADRATIC asymptotic tail — exactly the gelu family. Every
// other shape falls back to the untouched dispatch.
#if defined(DST_COEFF_STORE) && defined(EVAL_METHOD_POLY_CASCADE) && defined(EMBEDDED_LUT) && defined(USE_BF16) &&  \
    defined(ARCH_BLACKHOLE) && defined(USE_DUAL_EVAL) && !defined(POLY_PARITY_ODD) && !defined(POLY_PARITY_EVEN) && \
    !defined(BASIS_INPUT_ABS_X) && !defined(BASIS_MUL_ABS_X_BEFORE_POST) && !defined(BASIS_MUL_SQRT_1_MINUS_ABS) && \
    !defined(BASIS_AFFINE_EVEN) && !defined(BASIS_RIGHT_TAIL_ABS_AFFINE) && !defined(BASIS_CLAMP_MAX) &&            \
    !defined(BASIS_POST_SIGN_X) && !defined(BASIS_POST_REFLECT_PI) && !defined(BASIS_LEFT_TAIL_ZERO) &&             \
    !defined(BASIS_RIGHT_TAIL_IDENTITY) && !defined(RANGE_REDUCTION_EXP) && !defined(RANGE_REDUCTION_TRIG) &&       \
    !defined(RANGE_REDUCTION_TAN) && !defined(RANGE_REDUCTION_LOG) && !defined(RANGE_REDUCTION_CBRT) &&             \
    !defined(HAS_CRITICAL_POINT) && !defined(POSTCOMPOSE_AFFINE_Y) && !defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) &&     \
    !defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) && !defined(ASYMPTOTIC_FACTOR_X) && !defined(ASYMPTOTIC_FACTOR_QUADRATIC)
#define DST_COEFF_ELIGIBLE 1
#else
#define DST_COEFF_ELIGIBLE 0
#endif

#if DST_COEFF_ELIGIBLE

namespace dstcoeff {

constexpr uint32_t kCPS = POLY_DEGREE + 1;
constexpr uint32_t kCO = NUM_SEGMENTS + 1;

constexpr uint32_t seg_degree(uint32_t s) {
#ifdef HAS_SEGMENT_DEGREES
    return SEGMENT_DEGREES[s];
#else
    (void)s;
    return POLY_DEGREE;
#endif
}

#if defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) || defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD)
constexpr uint32_t kEvenMirrorFirstSegment = NUM_SEGMENTS / 2;
#if defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD)
constexpr uint32_t kEvenMirrorDeclaredSegments = TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_SEGMENTS;
#else
constexpr uint32_t kEvenMirrorDeclaredSegments = TT_PROGRAM_DST_COEFF_EVEN_MIRROR_SEGMENTS;
#endif

constexpr bool even_mirror_table_matches() {
    if ((NUM_SEGMENTS < 2) || (NUM_SEGMENTS & 1) || kEvenMirrorDeclaredSegments != NUM_SEGMENTS / 2) {
        return false;
    }
    for (uint32_t pair = 0; pair < NUM_SEGMENTS / 2; ++pair) {
        const uint32_t left = NUM_SEGMENTS / 2 - 1 - pair;
        const uint32_t right = NUM_SEGMENTS / 2 + pair;
        if (seg_degree(left) != seg_degree(right)) {
            return false;
        }
        const uint32_t left_lo = __builtin_bit_cast(uint32_t, LUT_DATA[left]);
        const uint32_t left_hi = __builtin_bit_cast(uint32_t, LUT_DATA[left + 1]);
        const uint32_t right_lo = __builtin_bit_cast(uint32_t, LUT_DATA[right]);
        const uint32_t right_hi = __builtin_bit_cast(uint32_t, LUT_DATA[right + 1]);
        if ((left_lo ^ right_hi) != 0x80000000u ||
            (((left_hi & 0x7fffffffu) != 0u) && ((left_hi ^ right_lo) != 0x80000000u)) ||
            (((left_hi & 0x7fffffffu) == 0u) && ((right_lo & 0x7fffffffu) != 0u))) {
            return false;
        }
        for (uint32_t k = 0; k <= seg_degree(left); ++k) {
            const uint32_t left_idx = kCO + left * kCPS + k;
            const uint32_t right_idx = kCO + right * kCPS + k;
            const uint32_t left_bits = __builtin_bit_cast(uint32_t, LUT_DATA[left_idx]);
            const uint32_t right_bits = __builtin_bit_cast(uint32_t, LUT_DATA[right_idx]);
            const uint32_t expected_xor = (k & 1) ? 0x80000000u : 0u;
            if ((left_bits ^ right_bits) != expected_xor) {
                return false;
            }
        }
    }
    return true;
}
static_assert(even_mirror_table_matches(), "even-mirror certificate disagrees with emitted selected coefficient table");

constexpr bool even_mirror_uses_lut_index(uint32_t idx) {
    if (idx < kCO) {
        return idx > kEvenMirrorFirstSegment && idx < NUM_SEGMENTS;
    }
    return ((idx - kCO) / kCPS) >= kEvenMirrorFirstSegment;
}
#else
constexpr bool even_mirror_uses_lut_index(uint32_t) { return true; }
#endif

// bf16-exact values materialize as ONE SFPLOADI (16-bit immediate) — parking
// them would trade 1 loadi for 1 load (wash) while paying park overhead.
// Keep them as immediates; park only the 2-loadi (full fp32) values.
constexpr bool bf16_exact(float f) { return (__builtin_bit_cast(uint32_t, f) & 0xFFFFu) == 0u; }

// Park enumeration order: boundaries lut[1..N-1], then per-segment coeffs
// lut[kCO + s*kCPS + k] for k = 0..seg_degree(s). Returns the park row for a
// LUT index, or -1 when the value stays an immediate.
constexpr int park_row_of_lut(uint32_t idx) {
    int row = 0;
    for (uint32_t s = 1; s < NUM_SEGMENTS; s++) {
        if (even_mirror_uses_lut_index(s) && !bf16_exact(LUT_DATA[s])) {
            if (s == idx) {
                return row;
            }
            row++;
        } else if (s == idx) {
            return -1;
        }
    }
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        for (uint32_t k = 0; k <= seg_degree(s); k++) {
            const uint32_t i = kCO + s * kCPS + k;
            if (even_mirror_uses_lut_index(i) && !bf16_exact(LUT_DATA[i])) {
                if (i == idx) {
                    return row;
                }
                row++;
            } else if (i == idx) {
                return -1;
            }
        }
    }
    return -1;
}

constexpr int lut_parked_count() {
    int row = 0;
    for (uint32_t s = 1; s < NUM_SEGMENTS; s++) {
        if (even_mirror_uses_lut_index(s) && !bf16_exact(LUT_DATA[s])) {
            row++;
        }
    }
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        for (uint32_t k = 0; k <= seg_degree(s); k++) {
            const uint32_t idx = kCO + s * kCPS + k;
            if (even_mirror_uses_lut_index(idx) && !bf16_exact(LUT_DATA[idx])) {
                row++;
            }
        }
    }
    return row;
}
constexpr int kNumLutParked = lut_parked_count();

// Asymptotic-tail constants (EXP_QUADRATIC only): same values, same order as
// asymptotic_exp / apply_asymptotic_scale in the sfpi path.
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
constexpr float kAsymVals[] = {
    1.4426950408889634f,       // [0] INV_LN2
    -0.6931152343750000f,      // [1] NEG_LN2_HI
    -3.19461832987e-05f,       // [2] NEG_LN2_LO
    1.0f / 120.0f,             // [3] Taylor c5
    1.0f / 24.0f,              // [4] Taylor c4
    1.0f / 6.0f,               // [5] Taylor c3
    ASYMPTOTIC_SCALE,          // [6]
    ASYMPTOTIC_BOUND,          // [7]
    ASYMPTOTIC_EXP_ARG_SCALE,  // [8]
};
constexpr int kNumAsymVals = 9;
constexpr int asym_row_of(int j) {
    int row = kNumLutParked;
    for (int i = 0; i < kNumAsymVals; i++) {
        if (!bf16_exact(kAsymVals[i])) {
            if (i == j) {
                return row;
            }
            row++;
        } else if (i == j) {
            return -1;
        }
    }
    return -1;
}
constexpr int asym_parked_count() {
    int n = 0;
    for (int i = 0; i < kNumAsymVals; i++) {
        if (!bf16_exact(kAsymVals[i])) {
            n++;
        }
    }
    return n;
}
constexpr int kNumAsymParked = asym_parked_count();
#else
constexpr int kNumAsymParked = 0;
#endif

constexpr int kTotalParked = kNumLutParked + kNumAsymParked;

}  // namespace dstcoeff

// Apply gate: multi-segment only (single-segment shapes belong to the TTI
// replay lowering), and the parked rows must fit the current half in
// FP32-ADDRESSABLE rows. An FP32-format dest access (dst_park_f32 /
// dst_load_f32) addresses 32-bit rows: one parked row consumes TWO 16-bit
// dest rows, so the 8-tile bf16 half (256 sixteen-bit rows) holds only 128
// fp32-addressable rows, data tile included — the same 128-row budget as
// the rational variant's fp32-dest half (kDstCoeffRowBase 32 + 96).
// The previous <= 256 bound let large cascades (polygamma p6_s16: 132,
// digamma p15_s32: 194) spill parked writes past the half into pack-owned
// rows: non-finite outputs from >= 145 tiles on the class-wide battery
// (the {145,146} ladder points exist for exactly this). Silicon-verified:
// shapes <= 128 pass the full C-VER-1 battery; over-capacity shapes now
// compile the untouched baseline (C-LOW-2).
constexpr bool kDstCoeffApply = (NUM_SEGMENTS >= 2) && (kDstCoeffBase + dstcoeff::kTotalParked <= 128);

#if defined(TT_INLINE_PROGRAM_DST_COEFF_LATE_RAW_SCHEDULE)
static_assert(kDstCoeffApply, "verified late raw-DST transport requires the active dense coefficient schedule");
#endif

namespace dstcoeff {

// --- per-tile park (values are compile-time constants -> LOADI pair + one
// --- FP32-format SFPSTORE each; rows rewritten EVERY tile because the pack
// --- scaffolding ZEROACCs the half after it drains: llk_pack_common.h
// --- _llk_pack_dest_section_done_ TT_ZEROACC(CLR_HALF)).
// C-VER-1 compile-time parking closure: assign boundaries, coefficients, and
// asymptotic constants directly to DST. Passing a vFloat through the
// runtime/probe helper creates an ABI boundary at which SFPI may attempt an
// illegal spill while expanding these recursive templates.
template <uint32_t S>
__attribute__((always_inline)) inline void park_boundaries() {
    if constexpr (S < NUM_SEGMENTS) {
        constexpr int row = park_row_of_lut(S);
        if constexpr (row >= 0) {
            dst_reg[kDstCoeffBase + row].mode<DataLayout::F32>() = vFloat(LUT_DATA[S]);
        }
        park_boundaries<S + 1>();
    }
}

template <uint32_t S, uint32_t K>
__attribute__((always_inline)) inline void park_seg_coeffs() {
    if constexpr (K <= seg_degree(S)) {
        constexpr uint32_t idx = kCO + S * kCPS + K;
        constexpr int row = park_row_of_lut(idx);
        if constexpr (row >= 0) {
            dst_reg[kDstCoeffBase + row].mode<DataLayout::F32>() = vFloat(LUT_DATA[idx]);
        }
        park_seg_coeffs<S, K + 1>();
    }
}

template <uint32_t S>
__attribute__((always_inline)) inline void park_coeffs() {
    if constexpr (S < NUM_SEGMENTS) {
        park_seg_coeffs<S, 0>();
        park_coeffs<S + 1>();
    }
}

#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
template <int J>
__attribute__((always_inline)) inline void park_asym() {
    if constexpr (J < kNumAsymVals) {
        constexpr int row = asym_row_of(J);
        if constexpr (row >= 0) {
            // Keep the compile-time SFPU value inside the destination-register
            // assignment.  Passing it through dst_park_f32(vFloat) gives the
            // compiler an ABI-visible object boundary in this recursive
            // template and can make it attempt an illegal vFloat spill.  The
            // value, row, and reload path are otherwise identical.
            dst_reg[kDstCoeffBase + row].mode<DataLayout::F32>() = vFloat(kAsymVals[J]);
        }
        park_asym<J + 1>();
    }
}
#endif

__attribute__((always_inline)) inline void park_all() {
    park_boundaries<1>();
    park_coeffs<0>();
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
    park_asym<0>();
#endif
}

// --- fetchers: parked -> single SFPLOAD (FP32 format); immediate otherwise.
template <uint32_t IDX>
__attribute__((always_inline)) inline vFloat lut_fetch() {
    constexpr int row = park_row_of_lut(IDX);
    if constexpr (row >= 0) {
        return dst_load_f32(kDstCoeffBase + row);
    } else {
        return vFloat(LUT_DATA[IDX]);
    }
}

#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
template <int J>
__attribute__((always_inline)) inline vFloat asym_fetch() {
    constexpr int row = asym_row_of(J);
    if constexpr (row >= 0) {
        return dst_load_f32(kDstCoeffBase + row);
    } else {
        return vFloat(kAsymVals[J]);
    }
}
#endif

// --- dual Horner over one segment (identical op sequence to
// --- eval_polynomial_dual: shared coefficient register, MAD(r, x, c) rungs).
// A verifier-bound selected reconstruction may request the single-row twin
// below when an auxiliary tail leaf would push the dual body's six persistent
// lane values beyond Blackhole's eight SFPU LREGs.  Both schedules use these
// exact parked fetchers, coefficient bytes, segment predicates, and Horner
// order; only cross-lane interleaving changes.
template <uint32_t BASE, int K>
__attribute__((always_inline)) inline void horner_rungs_single(vFloat x, vFloat& r) {
    if constexpr (K >= 0) {
        r = r * x + lut_fetch<BASE + (uint32_t)K>();
        horner_rungs_single<BASE, K - 1>(x, r);
    }
}

template <uint32_t SEG>
__attribute__((always_inline)) inline void eval_seg_single(vFloat x, vFloat& r) {
    constexpr uint32_t DEG = seg_degree(SEG);
    constexpr uint32_t BASE = kCO + SEG * kCPS;
    r = lut_fetch<BASE + DEG>();
    if constexpr (DEG > 0) {
        horner_rungs_single<BASE, (int)DEG - 1>(x, r);
    }
}

template <uint32_t SEG>
__attribute__((always_inline)) inline void cascade_single(vFloat x, vFloat& r) {
    if constexpr (SEG < NUM_SEGMENTS) {
        vFloat tmp;
        eval_seg_single<SEG>(x, tmp);
        v_if(x >= lut_fetch<SEG>()) { r = tmp; }
        v_endif;
        cascade_single<SEG + 1>(x, r);
    }
}

template <uint32_t BASE, int K>
__attribute__((always_inline)) inline void horner_rungs_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
    if constexpr (K >= 0) {
        vFloat c = lut_fetch<BASE + (uint32_t)K>();
        r1 = r1 * x1 + c;
        r2 = r2 * x2 + c;
        horner_rungs_dual<BASE, K - 1>(x1, x2, r1, r2);
    }
}

template <uint32_t SEG>
__attribute__((always_inline)) inline void eval_seg_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
    constexpr uint32_t DEG = seg_degree(SEG);
    constexpr uint32_t BASE = kCO + SEG * kCPS;
    vFloat c = lut_fetch<BASE + DEG>();
    r1 = c;
    r2 = c;
    if constexpr (DEG > 0) {
        horner_rungs_dual<BASE, (int)DEG - 1>(x1, x2, r1, r2);
    }
}

// --- segment cascade (identical predicated-move structure to
// --- unroll_segment_dual; boundary compare reads the parked row).
template <uint32_t SEG>
__attribute__((always_inline)) inline void cascade_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
    if constexpr (SEG < NUM_SEGMENTS) {
        vFloat tmp1, tmp2;
        eval_seg_dual<SEG>(x1, x2, tmp1, tmp2);
        vFloat b = lut_fetch<SEG>();
        v_if(x1 >= b) { r1 = tmp1; }
        v_endif;
        v_if(x2 >= b) { r2 = tmp2; }
        v_endif;
        cascade_dual<SEG + 1>(x1, x2, r1, r2);
    }
}

#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
// asymptotic_exp with parked constants — same arithmetic, same order.
__attribute__((always_inline)) inline vFloat asym_exp_dst(vFloat arg) {
    vFloat z = arg * asym_fetch<0>();  // INV_LN2
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    vInt k_int = as<vInt>(tmp) - as<vInt>(c231);
    vFloat r = k * asym_fetch<1>() + arg;  // NEG_LN2_HI
    r = k * asym_fetch<2>() + r;           // NEG_LN2_LO
    vFloat p = asym_fetch<3>();            // 1/120
    p = p * r + asym_fetch<4>();           // 1/24
    p = p * r + asym_fetch<5>();           // 1/6
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    // Same underflow guard as asymptotic_exp in piecewise_generic.cpp: a biased
    // exponent <= 0 wraps in setexp and returns ~1e38 instead of ~0.  Clamp
    // AFTER setexp (setexp rebuilds a nonzero value from a zero mantissa).
    vInt p_exp = exexp(p, ExponentMode::Biased);
    vInt new_exp = p_exp + k_int;
    vFloat out = setexp(p, new_exp);
    v_if(new_exp <= 0) { out = 0.0f; }
    v_endif;
    return out;
}

__attribute__((always_inline)) inline void asym_apply_dst(vFloat x_orig, vFloat& result) {
#if defined(ASYMPTOTIC_REGION_LEFT)
    v_if(x_orig < asym_fetch<7>()) {
#elif defined(ASYMPTOTIC_REGION_RIGHT)
    v_if(x_orig >= asym_fetch<7>()) {
#else  // ASYMPTOTIC_REGION_ALL
    {
#endif
        vFloat t = x_orig * x_orig * asym_fetch<8>();  // ARG_SCALE
        vFloat v = result * asym_exp_dst(t);
        v = v * asym_fetch<6>();  // ASYMPTOTIC_SCALE
#ifdef ASYMPTOTIC_NEGATE_OUTPUT
        v = as<vFloat>(as<vInt>(v) ^ (vInt)0x80000000);
#endif
        result = v;
    }
#if defined(ASYMPTOTIC_REGION_LEFT) || defined(ASYMPTOTIC_REGION_RIGHT)
    v_endif;
#endif
}
#endif  // ASYMPTOTIC_FACTOR_EXP_QUADRATIC

}  // namespace dstcoeff

// Main body: mirrors piecewise_generic_lut_specialized_N_dual for the gated
// shape (no basis, no range reduction) with parked coefficient fetches.
template <uint32_t POLY_DEG_T, uint32_t NUM_SEG_T, uint32_t LUT_SIZE_T>
inline void piecewise_generic_lut_dst_coeff(const std::array<float, LUT_SIZE_T>& lut) {
    (void)lut;
    dstcoeff::park_all();
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW)
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY)
    static_assert(
        TT_PROGRAM_DST_COEFF_DUAL_PEAK_LIVE > TT_PROGRAM_SFPU_LREG_CAPACITY,
        "single-row DST schedule requires a verifier-derived dual live-set overflow");
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE)
    static_assert(
        TT_PROGRAM_DST_COEFF_DUAL_PEAK_LIVE == 9u && TT_PROGRAM_AUX_LEAF_MAX_LIVE == 3u &&
            TT_PROGRAM_DST_COEFF_LATE_FINALIZER_MAX_LIVE == 3u,
        "same-row gradient finalization requires the exact verifier-derived peak-9 shape");
    static_assert(
        TT_PROGRAM_DST_COEFF_GRAD_FINALIZE_PEAK_LIVE == 2u &&
            TT_PROGRAM_DST_COEFF_GRAD_FINALIZE_PEAK_LIVE <= TT_PROGRAM_SFPU_LREG_CAPACITY,
        "same-row gradient finalizer exceeds the verified SFPU live-set capacity");
#if !defined(FUSE_GRAD_MUL)
#error "same-row gradient finalization requires the fused-gradient ABI"
#endif
#endif
#endif
#if defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) && !defined(FUSE_GRAD_MUL)
#error "typed same-row gradient finalization requires the fused-gradient ABI"
#endif
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare_raw_domain_input(x_raw);
        vFloat r;
#if defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) || defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD)
        vFloat x_folded = setsgn(x, 0);
        dstcoeff::eval_seg_single<dstcoeff::kEvenMirrorFirstSegment>(x_folded, r);
        dstcoeff::cascade_single<dstcoeff::kEvenMirrorFirstSegment + 1>(x_folded, r);
#else
        dstcoeff::eval_seg_single<0>(x, r);
        dstcoeff::cascade_single<1>(x, r);
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        dstcoeff::asym_apply_dst(x, r);
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) || defined(TT_INLINE_PROGRAM_RAW_ZERO_BOUNDARY) || \
    defined(TT_INLINE_PROGRAM_RESULT_TERMINAL_RAW_OVERRIDE)
        // Complete the ordinary overlay/reconstruction before materializing
        // encoded classifier state.  The three-argument overload preserves
        // this exact order for other schedules, while the already-selected
        // single-row schedule makes the lifetime boundary explicit so SFPI
        // cannot retain raw_u16 across the inlined reconstruction body.
        finalize_raw_domain_actions(x_raw, r);
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        finalize_encoded_raw_domain_actions(x_raw, raw_u16, r);
#else
        finalize_raw_domain_actions(x_raw, r);
#endif
#ifdef USE_BF16
        r = convert<vFloat16b>(r, RoundMode::Nearest);
#endif
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) || \
    defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE)
        // Preserve the ABI's existing double rounding exactly: the selected
        // derivative is first rounded to BF16, then multiplied by the BF16
        // incoming gradient parked in DST tile 1, then rounded again.  Loading
        // grad only here keeps it out of Horner/reconstruction liveness.
        vFloat grad = dst_reg[32 + d];
        r = r * grad;
#ifdef USE_BF16
        r = convert<vFloat16b>(r, RoundMode::Nearest);
#endif
#endif
        dst_reg[d] = r;
    }
#else
    for (int d = 0; d < 32; d += 2) {
        vFloat x_raw1 = dst_reg[d];
        vFloat x_raw2 = dst_reg[d + 1];
        vFloat x1 = prepare_raw_domain_input(x_raw1);
        vFloat x2 = prepare_raw_domain_input(x_raw2);
        vFloat r1, r2;
#if defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) || defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD)
        vFloat x_folded1 = setsgn(x1, 0);
        vFloat x_folded2 = setsgn(x2, 0);
        dstcoeff::eval_seg_dual<dstcoeff::kEvenMirrorFirstSegment>(x_folded1, x_folded2, r1, r2);
        dstcoeff::cascade_dual<dstcoeff::kEvenMirrorFirstSegment + 1>(x_folded1, x_folded2, r1, r2);
#else
        dstcoeff::eval_seg_dual<0>(x1, x2, r1, r2);
        dstcoeff::cascade_dual<1>(x1, x2, r1, r2);
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        dstcoeff::asym_apply_dst(x1, r1);
        dstcoeff::asym_apply_dst(x2, r2);
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) || defined(TT_INLINE_PROGRAM_RAW_ZERO_BOUNDARY) || \
    defined(TT_INLINE_PROGRAM_RESULT_TERMINAL_RAW_OVERRIDE)
        // Reload each preserved raw lane only after the dual selected Horner
        // and reconstruction temporaries have retired.  This retains the
        // existing coefficient residency and two-row evaluator schedule.
        vUInt raw_u16_1 = dst_reg[d].mode<DataLayout::U16>();
        vUInt raw_u16_2 = dst_reg[d + 1].mode<DataLayout::U16>();
        finalize_raw_domain_actions(x_raw1, raw_u16_1, r1);
        finalize_raw_domain_actions(x_raw2, raw_u16_2, r2);
#else
        finalize_raw_domain_actions(x_raw1, r1);
        finalize_raw_domain_actions(x_raw2, r2);
#endif
#ifdef USE_BF16
        r1 = convert<vFloat16b>(r1, RoundMode::Nearest);
        r2 = convert<vFloat16b>(r2, RoundMode::Nearest);
#endif
        dst_reg[d] = r1;
        dst_reg[d + 1] = r2;
    }
#endif
}

#endif  // DST_COEFF_ELIGIBLE
