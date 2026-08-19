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

// Batch-2 fixed dispatch scalars, shared with the golden and identical to the
// values the production dispatch sends (sfpu_operations.h / golden_generators
// _FMOD_DIVISOR/_REMAINDER_DIVISOR = 2.0, _UNARY_POWER_EXP = 2.0,
// _XIELU_ALPHA_P/_XIELU_ALPHA_N = 1.0).  Both legs must always receive
// identical values.
constexpr float FRESH_FMOD_DIVISOR           = 2.0f;
constexpr float FRESH_FMOD_DIVISOR_RECIP     = 0.5f;
constexpr std::uint32_t FRESH_POWER_EXPONENT = 0x40000000u; // 2.0f
constexpr std::uint32_t FRESH_XIELU_ALPHA    = 0x3f800000u; // 1.0f

// Batch-3 fixed dispatch scalars (production dispatch: softplus beta = 1.0,
// 1/beta = 1.0, threshold = 20.0 — shared with golden_generators
// _SOFTPLUS_BETA/_SOFTPLUS_THRESHOLD).
constexpr float FRESH_SOFTPLUS_BETA       = 1.0f;
constexpr float FRESH_SOFTPLUS_BETA_RECIP = 1.0f;
constexpr float FRESH_SOFTPLUS_THRESHOLD  = 20.0f;

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

// MIGRATED (storm contract, fresh_cpp/README.md — new bodies never land here):
//   calculate_tanh_fresh_cpp                -> fresh_cpp/tanh.h
//   calculate_tanh_derivative_lut_fresh_cpp -> fresh_cpp/tanhderivative-lut.h

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

// ---------------------------------------------------------------------------
// Lane BR causal-tier lift, batch 2.  Same doctrine as batch 1: each body
// states the production kernel's OWN algorithm (identical golden math
// constants) in plain row-at-a-time typed C++ — every constant a local, no
// programmed constant registers, no builtin MAD pinning, no hand interleave,
// no exponent-shift strength reductions where a plain statement exists.
// ---------------------------------------------------------------------------

// Shared: round-to-nearest integer and its int value via the 1.5*2^23
// rounding-bias identity (|z| < 2^22; golden math, the same identity the
// production expm1/exp kernels use through raw bit reads).
sfpi_inline sfpi::vFloat fresh_round_nearest(const sfpi::vFloat z, sfpi::vInt& k_int)
{
    constexpr float ROUNDING_BIAS = 12582912.0f; // 1.5 * 2^23
    const sfpi::vFloat t          = z + ROUNDING_BIAS;
    k_int                         = sfpi::as<sfpi::vInt>(t) - sfpi::as<sfpi::vInt>(sfpi::vFloat(ROUNDING_BIAS));
    return t - ROUNDING_BIAS;
}

// Shared: truncate-toward-zero via round-nearest + downward fixup on the
// magnitude (exact for every finite input; pass-through for |v| >= 2^23,
// inf, NaN — the same contract as the production kernels' exponent-shift
// truncation).
sfpi_inline sfpi::vFloat fresh_trunc_magnitude(const sfpi::vFloat v)
{
    constexpr float MANTISSA_SHIFT = 8388608.0f; // 2^23
    sfpi::vFloat r                 = v;
    sfpi::vFloat t                 = v + MANTISSA_SHIFT;
    t                              = t - MANTISSA_SHIFT;
    v_if (sfpi::exexp(v) < 23)
    {
        r = t;
    }
    v_endif;
    // Nearest may round up; truncation of a non-negative value never does.
    v_if (r > v)
    {
        r = r - 1.0f;
    }
    v_endif;
    return r;
}

// fmod / remainder core (production: metal ckernel_sfpu_fmod.h /
// ckernel_sfpu_remainder.h — divisor and reciprocal smuggled through
// vConstFloatPrgm0/1 by init, exponent-shift truncation, unroll-0 pins).
// Same algorithm: |v| minus trunc(|v|*recip)*s, the fixed residual mop-up,
// and the |v|==s zero snap; divisor/recip are the golden's fixed dispatch
// constants (2.0, 0.5) as plain locals.
sfpi_inline sfpi::vFloat fresh_fmod_core(const sfpi::vFloat v_mag, const sfpi::vFloat s, const sfpi::vFloat recip)
{
    sfpi::vFloat v        = v_mag;
    sfpi::vFloat quotient = fresh_trunc_magnitude(v * recip);
    v                     = v - quotient * s;

    // Residual mop-up (production-identical iteration count; value-bearing).
    constexpr int MOP_UP_ITERATIONS = 10;
    for (int l = 0; l < MOP_UP_ITERATIONS; ++l)
    {
        v_if (v >= s)
        {
            v = v - s;
        }
        v_endif;
    }
    return v;
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_fmod_fresh_cpp(const float divisor, const float divisor_recip)
{
    const sfpi::vFloat s     = sfpi::abs(sfpi::vFloat(divisor));
    const sfpi::vFloat recip = sfpi::abs(sfpi::vFloat(divisor_recip));
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v         = fresh_fmod_core(sfpi::abs(val), s, recip);
        // fmod keeps the dividend's sign.
        v = sfpi::copysgn(v, val);
        v_if (s == 0.0f)
        {
            v = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        v_if (sfpi::abs(v) - s == 0.0f)
        {
            v = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_remainder_fresh_cpp(const float divisor, const float divisor_recip)
{
    const sfpi::vFloat divisor_v = divisor;
    const sfpi::vFloat s         = sfpi::abs(divisor_v);
    const sfpi::vFloat recip     = sfpi::abs(sfpi::vFloat(divisor_recip));
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v         = fresh_fmod_core(sfpi::abs(val), s, recip);
        // remainder folds onto the divisor's sign (torch.remainder contract).
        v_if (val < 0.0f && v != 0.0f)
        {
            v = s - v;
        }
        v_endif;
        v_if (divisor_v < 0.0f && v != 0.0f)
        {
            v = v + divisor_v;
        }
        v_endif;
        v = sfpi::copysgn(v, divisor_v);
        v_if (s == 0.0f)
        {
            v = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        v_if (sfpi::abs(v) - s == 0.0f)
        {
            v = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Log, bf16 contract (production: tt_llk _calculate_log_body_ with the
// pre-shifted Chebyshev coefficients split across vConstFloatPrgm1/2 and
// inline literals).  Same executed constants (incl. the shipped B' = -0.7166
// the golden tolerance is fitted to), all local; ln(x) = poly(mantissa) +
// exponent * ln2; ln(0) = -inf.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_log_fresh_cpp()
{
    constexpr float A   = 0.1058f;
    constexpr float B   = -0.7166f;
    constexpr float C   = 2.0871f;
    constexpr float D   = -1.4753f;
    constexpr float LN2 = 0.692871f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat in = sfpi::dst_reg[0];
        const sfpi::vFloat x  = sfpi::setexp(in, 127); // mantissa into [1, 2)
        sfpi::vFloat series   = x * (x * (x * A + B) + C) + D;

        const sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(sfpi::exexp(in)), sfpi::RoundMode::Nearest);
        sfpi::vFloat result     = expf * LN2 + series;

        v_if (in == 0.0f)
        {
            result = -std::numeric_limits<float>::infinity();
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Expm1, bf16 contract (production: metal _sfpu_expm1_ non-fp32 branch —
// Juffa reduction with log2e / -ln2 / c1 parked in vConstFloatPrgm0/1/2, two
// raw __builtin_rvtt_sfpmad pins, and hand-interleaved Horner).  Identical
// arithmetic, plain statement: i = rint(a/ln2), f = a - i*ln2, quartic
// expm1(f), half-scaled 2^i reconstruction, saturation via the SMag8 clamp.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_expm1_fresh_cpp()
{
    constexpr float LOG2E   = 1.442695f;
    constexpr float NEG_LN2 = -0.6931471805599453f;
    constexpr float C3      = 8.361816406e-03f;
    constexpr float C2      = 4.177856445e-02f;
    constexpr float C1      = 1.666259766e-01f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vInt i;
        const sfpi::vFloat j = fresh_round_nearest(a * LOG2E, i);
        const sfpi::vFloat f = j * NEG_LN2 + a;

        sfpi::vFloat r       = C3;
        r                    = r * f + C2;
        r                    = r * f + C1;
        r                    = r * f + 0.5f;
        const sfpi::vFloat s = f * f;
        r                    = r * s + f;

        // For j == 0, r already is expm1(a); the half-scaled reconstruction
        // would flush tiny normal results through a subnormal.
        v_if (j != 0.0f)
        {
            const sfpi::vFloat w     = 0.5f;
            const sfpi::vFloat scale = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w)); // 0.5 * 2^i
            const sfpi::vFloat bias  = scale - w;
            const sfpi::vFloat jm2   = j + -2.0f;
            r                        = scale * r + bias;

            // Saturation: |i - 2| >= 127 covers a*log2(e) <= -125 / >= 129.
            const sfpi::vInt tail = sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag8>(sfpi::abs(jm2), sfpi::RoundMode::Nearest));
            v_if (tail >= 127)
            {
                // +inf on the positive side; NaN propagates through the multiply.
                r = jm2 * std::numeric_limits<float>::infinity();
                v_if (jm2 < 0.0f)
                {
                    r = -0.5f;
                }
                v_endif;
            }
            v_endif;
            r = r * 2.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

// MIGRATED (storm contract, fresh_cpp/README.md — new bodies never land here):
//   calculate_sqrt_rsqrt_fresh_cpp  -> fresh_cpp/sqrt.h (also serves rsqrt)
//   calculate_unary_power_fresh_cpp -> fresh_cpp/unarypower.h
//   calculate_xielu_fresh_cpp       -> fresh_cpp/xielu.h (natural loop-held-scalar form)

// ---------------------------------------------------------------------------
// Lane BR causal-tier lift, batch 3.  Same doctrine.
// ---------------------------------------------------------------------------

// Shared: exp(x) by the exp_21f exponent/mantissa recombination, clamped form
// (the production _sfpu_exp_21f_bf16_<true> contract: fp32 result, no bf16
// store rounding — callers own the store).  Same golden-math constants as
// calculate_exp_fresh_cpp; kept separate so the measured exp row's fresh body
// stays byte-stable.
sfpi_inline sfpi::vFloat fresh_exp_21f(const sfpi::vFloat val)
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    constexpr float C0      = 1.0017248f;
    constexpr float C1      = 7.839635491371155e-08f;
    constexpr float C2      = 4.791750143340323e-15f;

    sfpi::vFloat xlog2   = val * ONE_LN2 + 127.0f;
    xlog2                = sfpi::clamp(xlog2, 0.0f, 255.0f);
    const sfpi::vInt zi  = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
    const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
    frac              = (C2 * frac + C1) * frac + C0;
    return sfpi::setexp(frac, sfpi::exexp(z, sfpi::ExponentMode::Biased));
}

// Shared: reciprocal with Newton refinement, all constants literal (the
// production sfpu_reciprocal_iter reads its 2.0 from vConstFloatPrgm0 —
// the hand-ism these bodies remove).  Same NaN-by-sign-check contract.
template <int NEWTON_ITERATIONS>
sfpi_inline sfpi::vFloat fresh_recip(const sfpi::vFloat x)
{
    sfpi::vFloat y = sfpi::approx_recip(x);
    if constexpr (NEWTON_ITERATIONS > 0)
    {
        sfpi::vFloat t = x * y - 2.0f;
        if constexpr (NEWTON_ITERATIONS > 1)
        {
            const sfpi::vFloat y1 = y * -t - 0.0f;
            v_if (t < 0.0f)
            {
                t = x * y1 - 2.0f;
                y = y1 * -t - 0.0f;
            }
            v_endif;
        }
        else
        {
            v_if (t < 0.0f)
            {
                y = y * -t - 0.0f;
            }
            v_endif;
        }
    }
    return y;
}

// Sigmoid, bf16 non-approx contract (production: _sfpu_sigmoid_ spread across
// three headers — exp_21f helper + sfpu_reciprocal_iter<1> reading its 2.0
// from vConstFloatPrgm0).  sigmoid(x) = 1/(1 + exp(-x)) stated in one place,
// every constant local.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_sigmoid_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat e = fresh_exp_21f(-x);
        const sfpi::vFloat y = fresh_recip<1>(1.0f + e);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

// Cube root, bf16 contract (production: calculate_cube_root — Moroz magic
// seed via float-MAD-as-integer-divide with the refinement polynomial parked
// in vConstFloatPrgm0/1/2).  Identical algorithm, all constants local.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_cbrt_fresh_cpp()
{
    constexpr float NEG_THIRD_256 = -0x1.555556p-10f;
    constexpr float MAGIC         = 1418472267.0f / 256.0f + 8388608.0f; // 0x548c2b4b/256 + 2^23
    constexpr float Q0            = 0x1.c09806p0f;
    constexpr float Q1            = -0x1.403e6cp0f;
    constexpr float Q2            = 0x1.04cdb2p-1f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat a = sfpi::dst_reg[0];
        const sfpi::vFloat x = sfpi::abs(a);

        // Integer seed 0x548c2b4b - bits(x)/3 computed in float at 1/256 scale
        // (golden math: the paper's integer divide has no SFPU equivalent).
        sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(x), sfpi::RoundMode::Nearest);
        f              = f * NEG_THIRD_256 + MAGIC;
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(f) << 8);

        sfpi::vFloat dd      = x * (y * y);
        const sfpi::vFloat c = dd * y;
        const sfpi::vFloat t = c * (Q2 * c + Q1) + Q0;
        dd                   = sfpi::copysgn(dd, a);
        y                    = dd * (t * t);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

// MIGRATED (storm contract, fresh_cpp/README.md — new bodies never land here):
//   calculate_softplus_fresh_cpp -> fresh_cpp/softplus.h

// Hardsigmoid (production: calculate_activation reads slope/offset from
// vConstFloatPrgm0/1 programmed by hardsigmoid_init and clamps through the
// shared _relu_max_body_ helper).  hardsigmoid(x) = clamp(x/6 + 1/2, 0, 1)
// with both constants local; the same predicate order as the helper.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_hardsigmoid_fresh_cpp()
{
    constexpr float ONE_SIXTH = 0.1666666716337204f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat t       = v * ONE_SIXTH + 0.5f;
        v_if (t > 1.0f)
        {
            t = 1.0f;
        }
        v_endif;
        v_if (t < 0.0f)
        {
            t = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg++;
    }
}

// GELU, bf16 non-approx contract (production: calculate_gelu_piecewise —
// progressive v_and CC-narrowing inside one predicate block).  The same
// four-region piecewise CDF (identical constants, including the 2^-25
// ROUND_TO_GRID staircase snap, which is golden math reproducing torch's
// float32 erfc tail) stated as independent typed regions.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_gelu_fresh_cpp()
{
    constexpr float GELU_SAT         = -5.54259443f;
    constexpr float NEG_HALF_ONE_LN2 = -0.72134752044f; // -0.5 / ln(2)
    constexpr float HC0              = 3.0369991064e-01f;
    constexpr float HC1              = 9.5413386822e-02f;
    constexpr float HC2              = 1.3809983619e-02f;
    constexpr float HC3              = 7.5950479368e-04f;
    constexpr float ROUND_TO_GRID    = 0.375f;
    constexpr float E0               = 1.0017248f;
    constexpr float E1               = 7.839635491371155e-08f;
    constexpr float E2               = 4.791750143340323e-15f;
    constexpr float P0               = 5.000000000e-01f;
    constexpr float P1               = 3.9894227818e-01f;
    constexpr float P3               = -6.6361041488e-02f;
    constexpr float P5               = 9.7720050615e-03f;
    constexpr float P7               = -1.0717806322e-03f;
    constexpr float P9               = 8.1812159812e-05f;
    constexpr float P11              = -3.8082057209e-06f;
    constexpr float P13              = 7.9821413868e-08f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        // Identity region (x >= 2.78125) as the all-lane default.
        sfpi::vFloat r = x;
        v_if (x <= GELU_SAT)
        {
            r = 0.0f;
        }
        v_elseif (x < -3.125f)
        {
            // H = exp(-x^2/2) * corr_H(x), snapped to the 2^-25 grid.
            const sfpi::vFloat xlog2   = (x * x) * NEG_HALF_ONE_LN2 + 127.0f;
            const sfpi::vInt zi        = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
            const sfpi::vFloat z       = sfpi::as<sfpi::vFloat>(zi);
            sfpi::vFloat frac          = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            frac                       = (E2 * frac + E1) * frac + E0;
            const sfpi::vFloat exp_val = sfpi::setexp(frac, sfpi::exexp(z, sfpi::ExponentMode::Biased));

            const sfpi::vFloat H  = exp_val * (((HC3 * x + HC2) * x + HC1) * x + HC0);
            const sfpi::vFloat Hs = (H + ROUND_TO_GRID) - ROUND_TO_GRID;
            r                     = x * Hs;
        }
        v_elseif (x < 2.78125f)
        {
            const sfpi::vFloat x2 = x * x;
            sfpi::vFloat odd      = P13;
            odd                   = odd * x2 + P11;
            odd                   = odd * x2 + P9;
            odd                   = odd * x2 + P7;
            odd                   = odd * x2 + P5;
            odd                   = odd * x2 + P3;
            odd                   = odd * x2 + P1;
            r                     = x * (P0 + x * odd);
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

// Component-wise expm1 (production: tt-llk expm1_cw_clamped — Cody-Waite with
// the raw 0x4B400000 rounding-bias constant and the fused 0x4B3FFF81 ISUB;
// looped by a test adapter).  Same reduction, polynomials, and clamp; the
// round-nearest and the 2^k reconstruction stated typed.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_expm1_cw_fresh_cpp()
{
    constexpr float INV_LN2    = 1.4426950408889634f;
    constexpr float LN2_HI_NEG = -0.6931152343750000f;
    constexpr float LN2_LO_NEG = -3.19461832987e-05f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat x = sfpi::dst_reg[0];
        x              = sfpi::max(x, -87.0f);

        sfpi::vInt k_int;
        const sfpi::vFloat k = fresh_round_nearest(x * INV_LN2, k_int);
        sfpi::vFloat r       = k * LN2_HI_NEG + x;
        r                    = r + k * LN2_LO_NEG;

        // expm1(r) = r * h(r) (production Sollya fits per format arm).
#ifdef INP_FLOAT32
        sfpi::vFloat h = 1.3948583510e-03f;
        h              = h * r + 8.3691505715e-03f;
        h              = h * r + 4.1666239500e-02f;
        h              = h * r + 1.6666504741e-01f;
        h              = h * r + 5.0000000000e-01f;
        h              = h * r + 1.0f;
#else
        sfpi::vFloat h = 8.3751315251e-03f;
        h              = h * r + 4.1875664145e-02f;
        h              = h * r + 1.6666433215e-01f;
        h              = h * r + 4.9999371171e-01f;
        h              = h * r + 1.0f;
#endif
        h = r * h;

        const sfpi::vFloat two_k = sfpi::setexp(sfpi::vFloat(1.0f), k_int + 127);
        sfpi::dst_reg[0]         = (two_k - 1.0f) + two_k * h;
        sfpi::dst_reg++;
    }
}

// i1 (production: calculate_i1 — forced #pragma GCC unroll 1 and a
// compute-then-overwrite shape managing the SFPU register allocator, with the
// reciprocal's 2.0 in vConstFloatPrgm0).  Same rational/asymptotic algorithm
// and constants; the unroll pragma is dropped (unrolling is the compiler's)
// and every helper constant is literal.
inline sfpi::vFloat calculate_i1_asymptotic_fresh_cpp(const sfpi::vFloat abs_x, const sfpi::vFloat x_signed)
{
    // exp(|x|): |x| in [10, 88.5] precludes over/underflow, so the unclamped
    // recombination is exact here (the production kernel's _unsafe_ contract).
    constexpr float ONE_LN2    = 1.4426950216293334961f;
    constexpr float C0         = 1.0017248f;
    constexpr float C1         = 7.839635491371155e-08f;
    constexpr float C2         = 4.791750143340323e-15f;
    const sfpi::vFloat xlog2   = abs_x * ONE_LN2 + 127.0f;
    const sfpi::vInt zi        = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
    const sfpi::vFloat z       = sfpi::as<sfpi::vFloat>(zi);
    sfpi::vFloat frac          = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
    frac                       = (C2 * frac + C1) * frac + C0;
    const sfpi::vFloat exp_abs = sfpi::setexp(frac, sfpi::exexp(z, sfpi::ExponentMode::Biased));

    // 1/sqrt(|x|): the same SQRT_23 seed/coefficients as the fresh sqrt body.
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1));
    sfpi::vFloat c0      = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y              = rsqrt_y * (2.2825186f + c0 * (2.2533049f + c0));
    c0                   = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y              = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(1/|x|), degree-5 minimax (production constants).
    sfpi::vFloat correction = -3.3467922914e-01f;
    correction              = correction * inv_abs_x + -1.9748322314e-02f;
    correction              = correction * inv_abs_x + -4.3674591560e-02f;
    correction              = correction * inv_abs_x + -4.6652925320e-02f;
    correction              = correction * inv_abs_x + -1.4960495444e-01f;
    correction              = correction * inv_abs_x + 3.9894228967e-01f;

    return sfpi::copysgn(exp_abs * rsqrt_y * correction, x_signed);
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_i1_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat x           = sfpi::symmetric_clamp(sfpi::dst_reg[0], 88.5f);
        const sfpi::vFloat abs_x = sfpi::abs(x);

        // Rational path (valid for |x| <= 10), production constants per arm.
        sfpi::vFloat val;
        {
            const sfpi::vFloat t = x * x;
#ifdef INP_FLOAT32
            sfpi::vFloat number = 1.2293555930e-12f;
            number              = number * t + 7.7937084564e-10f;
            number              = number * t + 2.0916867527e-07f;
            number              = number * t + 2.8397364076e-05f;
            number              = number * t + 1.9247245509e-03f;
            number              = number * t + 5.6819390506e-02f;
            number              = number * t + 5.0000000000e-01f;
            sfpi::vFloat denom  = 7.4301498523e-19f;
            denom               = denom * t + -3.0635529988e-16f;
            denom               = denom * t + -3.1218170410e-13f;
            denom               = denom * t + 3.8127551116e-10f;
            denom               = denom * t + -1.9771712800e-07f;
            denom               = denom * t + 6.1268139689e-05f;
            denom               = denom * t + -1.1361218989e-02f;
            denom               = denom * t + 1.0f;
#else
            sfpi::vFloat number = 2.0223499130e-05f;
            number              = number * t + 1.6126291630e-03f;
            number              = number * t + 5.4503594600e-02f;
            number              = number * t + 4.9992737740e-01f;
            sfpi::vFloat denom  = -2.5076132990e-07f;
            denom               = denom * t + 1.0333660750e-04f;
            denom               = denom * t + -1.6242591070e-02f;
            denom               = denom * t + 1.0f;
#endif
            val = number * x * fresh_recip<2>(denom);
        }

        v_if (abs_x > 10.0f)
        {
            val = calculate_i1_asymptotic_fresh_cpp(abs_x, x);
        }
        v_endif;
#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
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
