// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#if TT_RATIONAL_DST_COEFF
static constexpr uint32_t kDstCoeffLutBase = NUM_SEGMENTS + 1;  // boundary points first
static constexpr uint32_t kDstCoeffCount = (NUM_DEGREE + DEN_DEGREE + 2u) * NUM_SEGMENTS;
// dst tile 1 (tile 0 holds the data), or tile 2 when FUSE_GRAD_MUL owns tile 1
// for the incoming grad. Same collision as
// dst_coeff_store.h::kDstCoeffBase -- see the comment there for the measured
// failure mode.
#if defined(FUSE_GRAD_MUL)
static constexpr uint32_t kDstCoeffRowBase = 64;
#else
static constexpr uint32_t kDstCoeffRowBase = 32;
#endif

// 0.0f / 1.0f ride the hardware constant LREGs (bitwise check: -0.0f must NOT
// match +0.0f — a -0.0 addend changes the sign of an exact-zero FMA result).
static constexpr bool dst_coeff_is_special(float v) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, v);
    return bits == 0x00000000u || bits == 0x3F800000u;
}

// Dense dst slot for coefficient j (LUT index kDstCoeffLutBase + j): number of
// non-special coefficients before j.
static constexpr uint32_t dst_coeff_slot(uint32_t j) {
    uint32_t s = 0;
    for (uint32_t k = 0; k < j; k++) {
        if (!dst_coeff_is_special(LUT_DATA[kDstCoeffLutBase + k])) {
            s++;
        }
    }
    return s;
}
static constexpr uint32_t kDstCoeffRows = dst_coeff_slot(kDstCoeffCount);
static_assert(
    kDstCoeffRowBase + kDstCoeffRows <= 128,
    "DST coefficient rows exceed fp32-dest half capacity (4 tiles)");

// Fetch coefficient j: literal for 0/1, single SFPLOAD from the parked dst row
// for everything else. The row index is a compile-time constant, so GCC emits
// one immediate-offset SFPLOAD (NOINC — the D-RWC walk is untouched).
template <uint32_t J>
static sfpi_inline vFloat dst_coeff_fetch() {
    constexpr float v = LUT_DATA[kDstCoeffLutBase + J];
    if constexpr (dst_coeff_is_special(v)) {
        return vFloat(v);
    } else {
        return dst_reg[kDstCoeffRowBase + dst_coeff_slot(J)];
    }
}

// Preload: 2xSFPLOADI + SFPSTORE per non-special coefficient (all lanes hold
// the broadcast value; runs outside any v_if so CC is all-on).
template <uint32_t J>
static inline void dst_coeff_preload_from() {
    if constexpr (J < kDstCoeffCount) {
        constexpr float v = LUT_DATA[kDstCoeffLutBase + J];
        if constexpr (!dst_coeff_is_special(v)) {
            dst_reg[kDstCoeffRowBase + dst_coeff_slot(J)] = vFloat(v);
        }
        dst_coeff_preload_from<J + 1>();
    }
}

// Descending Horner rung chain: acc = acc*x + c[I], I down to STOP inclusive.
template <uint32_t JBASE, int I, int STOP>
static sfpi_inline void dst_horner_chain(vFloat x, vFloat& acc) {
    if constexpr (I >= STOP) {
        acc = acc * x + dst_coeff_fetch<JBASE + static_cast<uint32_t>(I)>();
        dst_horner_chain<JBASE, I - 1, STOP>(x, acc);
    }
}

// Interleaved lockstep: both chains step down together (SFPMAD latency hiding,
// exactly the eval_rational_interleaved_numer_denom dataflow).
template <uint32_t NB, uint32_t DB, int I>
static sfpi_inline void dst_horner_lockstep(vFloat x, vFloat& numer, vFloat& denom) {
    if constexpr (I >= 0) {
        numer = numer * x + dst_coeff_fetch<NB + static_cast<uint32_t>(I)>();
        denom = denom * x + dst_coeff_fetch<DB + static_cast<uint32_t>(I)>();
        dst_horner_lockstep<NB, DB, I - 1>(x, numer, denom);
    }
}

// P(x), Q(x) for segment SEG — per-chain FMA sequence is EXACTLY that of
// eval_rational_interleaved_numer_denom (same values, same operation order,
// same bits); only the coefficient operand source changes.
template <uint32_t SEG>
static sfpi_inline void eval_rational_numer_denom_dst(vFloat x, vFloat& out_numer, vFloat& out_denom) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1u;
    constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1u;
    constexpr uint32_t NB = SEG * CPS;               // numerator coefficient base (j-space)
    constexpr uint32_t DB = NB + NUM_COEFFS;         // denominator coefficient base
    constexpr int MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    vFloat numer = dst_coeff_fetch<NB + NUM_DEGREE>();
    vFloat denom = dst_coeff_fetch<DB + DEN_DEGREE>();
    if constexpr (NUM_DEGREE > DEN_DEGREE) {
        dst_horner_chain<NB, static_cast<int>(NUM_DEGREE) - 1, static_cast<int>(DEN_DEGREE)>(x, numer);
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
        dst_horner_chain<DB, static_cast<int>(DEN_DEGREE) - 1, static_cast<int>(NUM_DEGREE)>(x, denom);
    }
    dst_horner_lockstep<NB, DB, MIN_DEG - 1>(x, numer, denom);
    out_numer = numer;
    out_denom = denom;
}

// Segment cascade, deferred reciprocal — the unroll_segment_rational_deferred
// twin (same v_if predication, boundary compares stay immediates).
template <uint32_t SEG>
static inline void dst_unroll_segments(vFloat x, vFloat& numer, vFloat& denom) {
    if constexpr (SEG < NUM_SEGMENTS) {
        v_if(x >= LUT_DATA[SEG]) { eval_rational_numer_denom_dst<SEG>(x, numer, denom); }
        v_endif;
        dst_unroll_segments<SEG + 1>(x, numer, denom);
    }
}

static inline void piecewise_rational_dst_coeff_tile(bool dst_preload) {
#if defined(TT_INLINE_PROGRAM_RATIONAL_LATE_RAW_SCHEDULE)
#error "raw BF16 rational suffix is incompatible with fp32-destination DST coefficients"
#endif
    if (dst_preload) {
        dst_coeff_preload_from<0>();
    }
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
#if defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
        vInt selected_core_exponent_delta;
        vFloat x = selected_core_exponent_replay_prepare(
            x_raw, selected_core_exponent_delta);
#else
        vFloat x = prepare_raw_domain_input(x_raw);
#endif
#if defined(TT_RATIONAL_COORDINATE_BOUND)
        x = rational_bound_coordinate(x);
#endif
        vFloat numer, denom;
        eval_rational_numer_denom_dst<0>(x, numer, denom);
        dst_unroll_segments<1>(x, numer, denom);
        // ONE deferred reciprocal, as the baseline path does.
        vFloat result = numer * rational_reciprocal(denom);
        // Gate requires USE_BF16: RNE narrowing before the store (the fp32
        // dest row then holds a bf16-representable value, so the packer's
        // fp32->bf16 conversion is exact — byte-identical to the bf16-dest
        // baseline's SFPSTORE truncation of the same rounded value).
#if defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
        selected_core_exponent_replay_finalize(
            x_raw, selected_core_exponent_delta, result);
#else
        finalize_raw_domain_actions(x_raw, result);
#endif
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[d] = result;
    }
}

#endif  // TT_RATIONAL_DST_COEFF
