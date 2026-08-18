// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only semantic implementations used by the corpus A/Bs.  These deliberately
// state the math one row at a time: no fixed LREGs, raw instructions, replay slots,
// SFPLOADMACRO templates, or hand-interleaved software schedule.
#include <cstdint>
namespace ckernel::sfpu {

template <int ITERATIONS>
inline void calculate_exp_fresh_cpp() {
    // exp(x) by exponent/mantissa recombination (the exp_21f algorithm of
    // Moroz et al. 2022, the same math as the production kernels), stated
    // row-at-a-time in plain vector C++:
    //
    //   exp(x) = 2**(x/ln2) = 2**xi * 2**xf.
    //
    // xlog2 = x/ln2 + 127 is the BIASED exponent of the result.  Feeding
    // xlog2 through mantissa-with-implicit-one << unbiased-exponent yields
    // the fixed-point encoding z whose exponent field is the integer part
    // and whose mantissa field is the fractional part.  The fractional
    // part is refined by the exp_21f quadratic and the two are recombined
    // with setexp.
    //
    // Range handling states what the math means at the boundaries:
    //  - above: exp overflows once the biased exponent would exceed 255,
    //    so xlog2 saturates at 255 (min against the bound);
    //  - below: exp underflows to zero once the biased exponent is not
    //    positive.  The recombination's exponent source is zeroed for
    //    those lanes; a zero exponent makes setexp produce a subnormal,
    //    which the bf16 store path flushes to zero.  The fractional
    //    refinement operates on the unmasked encoding: its value is
    //    irrelevant on underflowed lanes because the zero exponent alone
    //    forces the flush (the mantissa of a flushed subnormal is never
    //    observable).
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    constexpr float C0 = 1.0017248f;
    constexpr float C1 = 7.839635491371155e-08f;
    constexpr float C2 = 4.791750143340323e-15f;
    for (int row = 0; row < ITERATIONS; ++row) {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat xlog2 = val * ONE_LN2 + 127.0f;
        xlog2 = sfpi::min(xlog2, 255.0f);

        // Fixed-point encoding of xlog2: mantissa (implicit one) shifted
        // left by the unbiased exponent.
        sfpi::vInt iexp = sfpi::exexp(xlog2);
        sfpi::vInt zi = sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne);
        zi = sfpi::shft(zi, iexp, sfpi::ShiftMode::Logical);
        const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);

        // Quadratic refinement of 2**xf on [0, 1) from the unmasked
        // encoding's mantissa field.
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
        frac = (C2 * frac + C1) * frac + C0;

        // Underflow: zero the exponent source where xlog2 is not positive.
        sfpi::vFloat zc = z;
        v_if (xlog2 <= 0.0f) { zc = 0.0f; }
        v_endif;

        sfpi::vFloat y = sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));

        // bf16 destination: round to nearest-even before the store
        // truncates (keeps e.g. exp(ln(81)) at 81 rather than 80.5).
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
// Keep the semantic loop isolated so compiler A/Bs cannot inherit opaque setup,
// profiler, or counter ownership from the caller.
__attribute__((noinline)) void calculate_sigmoid_appx_fresh_cpp() {
    for (int row = 0; row < ITERATIONS; ++row) {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        const sfpi::vFloat input_squared = input * input;
        const sfpi::vFloat result = (-0.00447352f * input_squared + 0.19833094f) * input + 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
// The same SigmoidAppx contract stated as a 3-range magnitude dispatch tree
// with constant affine leaves -- the dataflow shape the compiler's LUT
// instruction selection (-mtt-tensix-optimize-lut-select) proves and
// re-selects as a single SFPLUTFP32.  Kept alongside the cubic above as a
// second independently measurable selector: coefficients are a 3-piece fit
// of sigmoid(|x|)-0.5 over the test's [-5, 5] stimulus domain; the explicit
// setsgn restores the odd symmetry outside the tree (SGN_RETAIN folding is a
// later compiler increment).  Same golden and tolerance contract as the
// cubic (exact torch.sigmoid, atol 0.13 / rtol 0.05).
__attribute__((noinline)) void calculate_sigmoid_appx_tree_cpp()
{
    for (int row = 0; row < ITERATIONS; ++row)
    {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        const sfpi::vFloat mag   = sfpi::abs(input);
        sfpi::vFloat g           = mag * 0.0375f + 0.3058f;
        v_if (mag < 1.0f)
        {
            g = mag * 0.2452f + -0.0005f;
        }
        v_elseif (mag < 2.0f)
        {
            g = mag * 0.1497f + 0.0814f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::setsgn(g, input) + 0.5f;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_signbit_fresh_cpp() {
    for (int row = 0; row < ITERATIONS; ++row) {
        // Keep the all-lane predicate boundary typed and local to this
        // out-of-line semantic body so the compiler can prove its CC state.
        __builtin_rvtt_sfppushc(0);
        __builtin_rvtt_sfppopc(0);
        const sfpi::vFloat input = sfpi::dst_reg[0];
        const sfpi::vInt sign = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(input), -31));
        sfpi::dst_reg[0] = sfpi::int32_to_float(sign, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

template <bool IS_MAX, int ITERATIONS>
__attribute__((noinline)) void calculate_binary_max_min_fresh_cpp() {
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face) {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row) {
            // Keep the all-lane predicate boundary typed and local.  Compiler
            // macro formation may hoist the identical enables with its owned
            // descriptor configuration after proving that this body has no
            // CC effects.
            __builtin_rvtt_sfppushc(0);
            __builtin_rvtt_sfppopc(0);
            const sfpi::vFloat lhs = sfpi::dst_reg[0];
            const sfpi::vFloat rhs = sfpi::dst_reg[tile_rows];
            sfpi::dst_reg[0] = IS_MAX ? sfpi::max(lhs, rhs) : sfpi::min(lhs, rhs);
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, bool IS_MAX, int ITERATIONS>
inline void call_binary_max_min_fresh_cpp(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out,
    const VectorMode vector_mode) {
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(
        dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh max/min expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh max/min expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh max/min expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper.  The isolated semantic
    // body then contains only constant relative Dst addresses, which are
    // representable in a compiler-owned macro descriptor.
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_binary_max_min_fresh_cpp<IS_MAX, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

template <bool IS_MAX, int ITERATIONS>
__attribute__((noinline)) void calculate_unary_max_min_fresh_cpp(const std::uint32_t value)
{
    // The scalar operand arrives as raw float bits, exactly as the production
    // kernel receives it; the typed body only names the semantic comparison.
    const sfpi::vFloat operand = Converter::as_float(value);

#pragma GCC unroll 8
    for (int row = 0; row < ITERATIONS; ++row)
    {
        // Keep the all-lane predicate boundary typed and local.  Compiler
        // macro formation may hoist the identical enables with its owned
        // descriptor configuration after proving that this body has no
        // CC effects.
        __builtin_rvtt_sfppushc(0);
        __builtin_rvtt_sfppopc(0);
        const sfpi::vFloat input = sfpi::dst_reg[0];
        sfpi::dst_reg[0]         = IS_MAX ? sfpi::max(input, operand) : sfpi::min(input, operand);
        sfpi::dst_reg++;
    }
}

template <bool IS_MAX, bool IS_UNSIGNED, int ITERATIONS>
__attribute__((noinline)) void calculate_unary_max_min_int_fresh_cpp(const std::uint32_t value)
{
#pragma GCC unroll 8
    for (int row = 0; row < ITERATIONS; ++row)
    {
        __builtin_rvtt_sfppushc(0);
        __builtin_rvtt_sfppopc(0);
        if constexpr (IS_UNSIGNED)
        {
            // Typed unsigned min/max; the SFPI library owns the MSB-safe
            // sign-magnitude lowering, the body only states the comparison.
            const sfpi::vUInt input = sfpi::dst_reg[0];
            sfpi::dst_reg[0]        = IS_MAX ? sfpi::max(input, value) : sfpi::min(input, value);
        }
        else
        {
            sfpi::vInt input         = sfpi::dst_reg[0];
            const sfpi::vInt operand = static_cast<int>(value);
            if constexpr (IS_MAX)
            {
                v_if (input < operand)
                {
                    input = operand;
                }
                v_endif;
            }
            else
            {
                v_if (input > operand)
                {
                    input = operand;
                }
                v_endif;
            }
            sfpi::dst_reg[0] = input;
        }
        sfpi::dst_reg++;
    }
}

// Fresh typed-C++ integer add/sub over the sign-magnitude Int32 Dst the binary
// harness drives (production dispatch: _add_int_/_sub_int_ with
// InstrModLoadStore::INT32 and SIGN_MAGNITUDE_FORMAT=true).  On Blackhole that
// production path is raw hand-scheduled TTI (TT_SFPLOAD + SFPCAST/SFPSETSGN
// sign-magnitude<->2's-complement conversions + TTI_SFPIADD + TT_SFPSTORE).
// This body states the same semantics in plain typed sfpi: DataLayout::SM32
// loads/stores carry the representation contract and the compiler owns
// conversion lowering, scheduling, and delivery.  No fixed LREGs, raw
// instructions, replay slots, or SFPLOADMACRO templates.
template <bool IS_ADD, int ITERATIONS>
__attribute__((noinline)) void calculate_add_sub_int_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt a                              = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt b                              = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = IS_ADD ? a + b : a - b;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, bool IS_ADD, int ITERATIONS>
inline void call_add_sub_int_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh add/sub int expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh add/sub int expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh add/sub int expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (same idiom as the fresh
    // max/min selector).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_add_sub_int_fresh_cpp<IS_ADD, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

// Fresh typed-C++ int32 multiply over the sign-magnitude Int32 Dst the binary
// harness drives.  The production kernel (metal ckernel_sfpu_mul_int32.h) reads
// the Dst bits raw (plain InstrModLoadStore::INT32, no representation
// conversion), which matches the golden's signed low-32 product only where
// sign-magnitude and two's-complement coincide -- the non-negative test domain.
// This body states the golden contract directly: DataLayout::SM32 loads/stores
// carry the representation contract (the compiler owns the conversion lowering
// -- a single self-inverse SFPCAST on BH), and the multiply is the plain
// radix-23 split identity over the typed 24x24 primitive (fractional_mul),
// retaining exactly the terms that contribute modulo 2^32 -- the same identity
// the handwritten kernel schedules by hand through SFPLOADMACRO templates.
// Signed multiplication has the same low 32 bits as unsigned multiplication
// over the same two's-complement bits, so the wrap product is computed on the
// raw bit patterns.  No fixed LREGs, raw instructions, replay slots, or
// SFPLOADMACRO templates.
#if !(__riscv_xtttensixbh || __riscv_xtttensixqsr)
// WH has no integer multiply instruction.  Radix 2^10 keeps every chunk
// product and coefficient below 2^23, so FP32 arithmetic and the 2^23
// mantissa-extraction conversion are exact (no saturation).
template <unsigned SHIFT_A, unsigned SHIFT_B>
inline sfpi::vFloat mul_int_fresh_cpp_chunk_product(const sfpi::vUInt a, const sfpi::vUInt b)
{
    constexpr unsigned mask  = 0x3ff;
    const sfpi::vInt a_chunk = sfpi::as<sfpi::vInt>((a >> SHIFT_A) & mask);
    const sfpi::vInt b_chunk = sfpi::as<sfpi::vInt>((b >> SHIFT_B) & mask);
    return sfpi::convert<sfpi::vFloat>(a_chunk, sfpi::RoundMode::Nearest) * sfpi::convert<sfpi::vFloat>(b_chunk, sfpi::RoundMode::Nearest);
}
#endif

template <int ITERATIONS>
__attribute__((noinline)) void calculate_mul_int_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

    // WH's 14-chunk-product emulation body is too large for a 4x face unroll
    // (TRISC1_CODE overflow); keep the rolled face loop there.  BH keeps the
    // add/sub-precedent shape.
#if __riscv_xtttensixbh || __riscv_xtttensixqsr
#pragma GCC unroll 4
#endif
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            // Keep the all-lane predicate boundary typed and local (the
            // fresh max/min precedent): compiler macro formation may hoist
            // the identical enables with its owned descriptor configuration
            // after proving this body has no CC effects.
            __builtin_rvtt_sfppushc(0);
            __builtin_rvtt_sfppopc(0);
            const sfpi::vInt a   = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt b   = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            const sfpi::vUInt ua = sfpi::as<sfpi::vUInt>(a);
            const sfpi::vUInt ub = sfpi::as<sfpi::vUInt>(b);
#if __riscv_xtttensixbh || __riscv_xtttensixqsr
            sfpi::vUInt lo = sfpi::fractional_mul(ua, ub, sfpi::FractionalHalf::Low);
            sfpi::vUInt hi = sfpi::fractional_mul(ua, ub, sfpi::FractionalHalf::High);
            hi += sfpi::fractional_mul(ua >> 23, ub, sfpi::FractionalHalf::Low);
            hi += sfpi::fractional_mul(ua, ub >> 23, sfpi::FractionalHalf::Low);
            const sfpi::vUInt product = lo + (hi << 23);
#else
            constexpr float bias = 8388608.0f; // 2^23

            // Build and consume one radix coefficient at a time so live SFPU
            // values stay below the eight-register architectural file.
            sfpi::vUInt product      = sfpi::exman(mul_int_fresh_cpp_chunk_product<0, 0>(ua, ub) + bias);
            sfpi::vFloat coefficient = mul_int_fresh_cpp_chunk_product<0, 10>(ua, ub) + mul_int_fresh_cpp_chunk_product<10, 0>(ua, ub) + bias;
            product += sfpi::vUInt(sfpi::exman(coefficient)) << 10;
            coefficient = mul_int_fresh_cpp_chunk_product<0, 20>(ua, ub) + mul_int_fresh_cpp_chunk_product<10, 10>(ua, ub) +
                          mul_int_fresh_cpp_chunk_product<20, 0>(ua, ub) + bias;
            product += sfpi::vUInt(sfpi::exman(coefficient)) << 20;
            coefficient = mul_int_fresh_cpp_chunk_product<0, 30>(ua, ub) + mul_int_fresh_cpp_chunk_product<10, 20>(ua, ub) +
                          mul_int_fresh_cpp_chunk_product<20, 10>(ua, ub) + mul_int_fresh_cpp_chunk_product<30, 0>(ua, ub) + bias;
            product += sfpi::vUInt(sfpi::exman(coefficient)) << 30;
#endif
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = sfpi::as<sfpi::vInt>(product);
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_mul_int_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh mul int expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh mul int expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh mul int expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (same idiom as the fresh
    // max/min and add/sub selectors).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_mul_int_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

template <bool DST_ACCUM_MODE, DataFormat FORMAT, int ITERATIONS>
inline void calculate_addcmul_fresh_cpp(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out,
    const std::uint32_t value) {
    static_assert(
        FORMAT == DataFormat::Float32 || FORMAT == DataFormat::Float16_b || FORMAT == DataFormat::Bfp8_b);
    constexpr std::uint32_t tile_rows = 32;
    const sfpi::vFloat scale = Converter::as_float(value);

#pragma GCC unroll 8
    for (int row = 0; row < ITERATIONS; ++row) {
        const sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * tile_rows];
        const sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * tile_rows];
        const sfpi::vFloat c = sfpi::dst_reg[dst_index_in2 * tile_rows];
        sfpi::vFloat result = (scale * b) * c + a;
        if constexpr (!DST_ACCUM_MODE) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[dst_index_out * tile_rows] = result;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------
// Lane BR causal-tier lift: fresh typed semantic bodies for rows whose
// production body is hand-shaped (raw TTI streams, fixed-LREG pinning, LUT
// l_reg idioms, hand software pipelines, programmed constant registers).
// Each body states the SAME golden contract as the production kernel it is
// paired with (identical constants where the constants are the golden math)
// in plain row-at-a-time typed C++: no fixed LREGs, raw instructions, replay
// slots, SFPLOADMACRO templates, or hand-interleaved schedules.
// ---------------------------------------------------------------------------

// Fixed clamp/hardtanh bounds shared with the golden and with the production
// dispatch (helpers/sfpu_dispatch_constants.py CLAMP_MIN/CLAMP_MAX; the
// production legs receive the same values as fp16/bf16 bit patterns).  The
// production and fresh legs must always receive identical bounds.
constexpr float FRESH_CLAMP_LO = -1.0f;
constexpr float FRESH_CLAMP_HI = 1.0f;

// Ceil (production: raw-TTI _ceil_body_ over l_reg-pinned _trunc_body_,
// tt_llk_blackhole ckernel_sfpu_rounding_ops.h).  Semantic statement: round
// to nearest via the 2^23 mantissa-shift (the clean idiom the same file's
// _round_even_ already uses), then bump lanes that rounded below the input.
// Exact for every finite input: values with exponent >= 23 have no fraction
// and keep their bits.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_ceil_fresh_cpp()
{
    constexpr float MANTISSA_SHIFT = 8388608.0f; // 2^23
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = v;
        // |v| + 2^23 - 2^23 rounds away the fraction (nearest-even) for all
        // |v| < 2^23; larger magnitudes (and inf/NaN) keep r = v below.
        sfpi::vFloat t = sfpi::abs(v) + MANTISSA_SHIFT;
        t              = t - MANTISSA_SHIFT;
        v_if (sfpi::exexp(v) < 23)
        {
            r = sfpi::copysgn(t, v);
        }
        v_endif;
        // Nearest-integer below the input means r = floor(v); ceil = r + 1.
        v_if (r < v)
        {
            r = r + 1.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

// EqualZero (production: fully raw-TTI calculate_comp float path with fixed
// LREG0/2/5 and an ADDR_MOD_6-fused store increment, metal ckernel_sfpu_comp.h).
// Semantic statement: 1.0 where |v| == 0 (covers -0.0; NaN keeps a nonzero
// magnitude and stays 0.0 — the production kernel's documented contract).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_eqz_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = 0.0f;
        v_if (sfpi::abs(v) == 0.0f)
        {
            r = 1.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

// Clamp (production: _calculate_clamp_ with fp16-bit-punned scalar params, a
// trailing offset addend, and #pragma GCC unroll 0).  The dispatch constants
// are the golden's CLAMP_MIN/CLAMP_MAX with offset 0, so the semantic
// statement is the bare clamp with typed float bounds.  The predicate form
// (not min/max) preserves the production kernel's NaN pass-through.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_clamp_fresh_cpp(const float lo, const float hi)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v < lo)
        {
            v = lo;
        }
        v_elseif (v >= hi)
        {
            v = hi;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Hardtanh (production: _calculate_hardtanh_ encodes the clamp as three
// chained add-then-zero-select steps over host-pre-negated bf16 params).
// hardtanh(x) = clamp(x, lo, hi) — same golden, same bounds; stated directly.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_hardtanh_fresh_cpp(const float lo, const float hi)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v < lo)
        {
            v = lo;
        }
        v_elseif (v >= hi)
        {
            v = hi;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Tanh, bf16 non-approx contract (production: _sfpu_tanh_polynomial_x2_ — an
// explicit two-datum hand software pipeline with three coefficients parked in
// programmed constant registers and a scalar epilogue).  Same Sollya
// polynomial (the golden math), one datum per row, every coefficient a plain
// local: pipelining, unrolling, and constant residency are the compiler's.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_fresh_cpp()
{
    constexpr float C1 = 0.999004364013671875f;
    constexpr float C2 = 3.0897438526153564453125e-2f;
    constexpr float C3 = -0.4890659749507904052734375f;
    constexpr float C4 = 0.281917631626129150390625f;
    constexpr float C5 = -6.6649019718170166015625e-2f;
    constexpr float C6 = 5.876733921468257904052734375e-3f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat r       = C6;
        r                    = r * a + C5;
        r                    = r * a + C4;
        r                    = r * a + C3;
        r                    = r * a + C2;
        r                    = r * a + C1;
        r                    = r * a;
        r                    = sfpi::min(r, 1.0f);
        r                    = sfpi::copysgn(r, x);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

// Tanh-derivative, legacy LUT contract (production: _calculate_tanh_derivative_
// pins l_reg[LReg0..2] across the tile and consumes the raw SFPLUT programmed
// by tanh_derivative_init's TT_SFPLOADI words).  The row's golden IS the
// 3-region piecewise-linear tanh (breakpoints 1.0/2.0, slopes 0.90625 and
// 0.09375x+0.8125, saturation 1.0), so the faithful semantic statement is the
// same piecewise dataflow as typed v_if regions (the sigmoidappx-tree
// precedent), then 1 - t^2.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_tanh_derivative_lut_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat a = sfpi::abs(x);
        sfpi::vFloat t       = 1.0f;
        v_if (a < 1.0f)
        {
            t = a * 0.90625f;
        }
        v_elseif (a < 2.0f)
        {
            t = a * 0.09375f + 0.8125f;
        }
        v_endif;
        sfpi::dst_reg[0] = t * (-t) + 1.0f;
        sfpi::dst_reg++;
    }
}

// Silu (production: _calculate_silu_ over the POLYVAL5 text macro with an
// abs/1-x symmetry fold; the row measures causal exactly 0.0% — the passes
// never engage the production structure).  Identical piecewise sigmoid math
// (the golden tolerance is fitted to it), restated with plain locals and a
// free loop so the compiler owns unrolling and delivery.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_silu_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v   = sfpi::dst_reg[0];
        const sfpi::vFloat mag = sfpi::abs(v);
        sfpi::vFloat sig       = 1.0f;
        v_if (mag <= 1.0f)
        {
            sig = mag * 0.229f + 0.5f;
        }
        v_elseif (mag < 5.0f)
        {
            sig = (((0.00144462f * mag + -0.01055479f) * mag + -0.01203685f) * mag + 0.24300185f) * mag + 0.50437757f;
        }
        v_endif;
        v_if (v < 0.0f)
        {
            sig = 1.0f - sig;
        }
        v_endif;
        sfpi::dst_reg[0] = v * sig;
        sfpi::dst_reg++;
    }
}

// Binary left shift, Int32 (production: calculate_binary_left_shift — an
// entirely raw TT_SFPLOAD/TTI_SFP* stream over fixed LREG0..4 with magic
// immediates 0xFE0/-32 and 0x020/32; the row is UNENGAGED at the current
// pins).  Semantic statement over the same INT32_2S_COMP load/store contract
// (typed DataLayout::SM32, the fresh add/sub/mul precedent): shift left by
// the per-lane amount, zero where the amount is outside [0, 32).
template <int ITERATIONS>
__attribute__((noinline)) void calculate_left_shift_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt value  = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt amount = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt result       = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(value), amount));
            v_if (amount < 0 || amount >= 32)
            {
                result = 0;
            }
            v_endif;
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = result;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_left_shift_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh left shift expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh left shift expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh left shift expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (the fresh max/min, add/sub,
    // and mul precedent).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_left_shift_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel::sfpu
