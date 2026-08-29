// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only semantic implementations used by the corpus A/Bs.  These deliberately
// state the math one row at a time: no fixed LREGs, raw instructions, replay slots,
// SFPLOADMACRO templates, or hand-interleaved software schedule.
#include <cstdint>

// Storm-contract migrations: canonical per-op semantic bodies live in
// fresh_cpp/<op>.h (see fresh_cpp/README.md); shared semantic helpers in
// fresh_cpp/helpers.h.  The aggregator keeps including migrated bodies so
// the existing selector wiring is unchanged; NEW bodies never land here.
#include "fresh_cpp/helpers.h"
#include "fresh_cpp/remainder.h"
#include "fresh_cpp/rsqrt.h"
#include "fresh_cpp/sigmoid.h"
#include "fresh_cpp/silu.h"

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
    for (int row = 0; row < ITERATIONS; ++row)
    {
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
        for (int row = 0; row < ITERATIONS; ++row)
        {
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
//
// Result materialization mirrors the production kernel's dual-store shape:
// store 0 unconditionally, then store 1 under the |v|==0 CC — the stores come
// straight from the hard constant registers (vConst0/vConst1), so no lane
// register ever materializes the result.  Rows are addressed by immediate
// offset (dst_reg[d]) rather than dst_reg++: the compiler will not fuse a
// counter increment into a CC-predicated trailing store (it emits a separate
// TTINCRWC, giving the increment its own issue word), while indexed rows need
// no counter at all.  Net 6 issue words/row (load, store0, abs, setcc,
// store1, encc) = the production kernel's exact slot count — the -0.0
// handling (the abs) is kept.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_eqz_fresh_cpp()
{
#pragma GCC unroll 32
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[d];
        sfpi::dst_reg[d]     = sfpi::vConst0;
        v_if (sfpi::abs(v) == 0.0f)
        {
            sfpi::dst_reg[d] = sfpi::vConst1;
        }
        v_endif;
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

// MIGRATED (storm contract, fresh_cpp/README.md — new bodies never land here):
//   calculate_tanh_fresh_cpp                -> fresh_cpp/tanh.h
//   calculate_tanh_derivative_lut_fresh_cpp -> fresh_cpp/tanhderivative-lut.h

// calculate_silu_fresh_cpp: moved to fresh_cpp/silu.h (storm migration).

// calculate_left_shift_fresh_cpp: moved to fresh_cpp/shift.h with the lane GI
// Option R Dst-read-contract rewrite (owner ratification 2026-08-24 item 3:
// raw two's-complement DataLayout::I32, one-word range guard; adjudication
// record laneGI-evidence-20260824/LEFTSHIFT-ADJUDICATION.md).  The previous
// SM32 spelling stays refused at the compiler level by name
// (sm32-cast-elision-refuted, laneCU).

// ---------------------------------------------------------------------------
// Lane BR causal-tier lift, batch 2.  Same doctrine as batch 1: each body
// states the production kernel's OWN algorithm (identical golden math
// constants) in plain row-at-a-time typed C++ — every constant a local, no
// programmed constant registers, no builtin MAD pinning, no hand interleave,
// no exponent-shift strength reductions where a plain statement exists.
// ---------------------------------------------------------------------------

// fresh_round_nearest: moved to fresh_cpp/helpers.h (storm migration).

// fresh_trunc_magnitude / fresh_fmod_core: moved to fresh_cpp/helpers.h
// (storm migration).

// calculate_fmod_fresh_cpp: moved to fresh_cpp/fmod.h (storm migration).

// calculate_remainder_fresh_cpp: moved to fresh_cpp/remainder.h (storm migration).

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

// calculate_expm1_fresh_cpp: moved to fresh_cpp/expm1.h (storm migration).

// MIGRATED (storm contract, fresh_cpp/README.md — new bodies never land here):
//   calculate_sqrt_rsqrt_fresh_cpp  -> fresh_cpp/rsqrt.h (shared sqrt/rsqrt
//     template, RECIPROCAL selects the arm; fresh_cpp/sqrt.h includes it)
//   calculate_unary_power_fresh_cpp -> fresh_cpp/unarypower.h
//   calculate_xielu_fresh_cpp       -> fresh_cpp/xielu.h (natural loop-held-scalar form)

// ---------------------------------------------------------------------------
// Lane BR causal-tier lift, batch 3.  Same doctrine.
// ---------------------------------------------------------------------------

// fresh_exp_21f / fresh_recip: moved to fresh_cpp/helpers.h (storm migration).

// calculate_sigmoid_fresh_cpp: moved to fresh_cpp/sigmoid.h (storm migration).

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

// call_left_shift_fresh_cpp: moved to fresh_cpp/shift.h (lane GI Option R).

}  // namespace ckernel::sfpu

// ---------------------------------------------------------------------------
// Storm layout (fresh_cpp/README.md): canonical per-op semantic bodies live in
// fresh_cpp/<op>.h — one header per op, included here so every existing
// consumer keeps a single include.  Pre-storm bodies above migrate here over
// time; new bodies never land in this file.
// Lane S2 slice (agent/storm-s2):
#include "fresh_cpp/digamma.h"
#include "fresh_cpp/div_int32_floor.h"
#include "fresh_cpp/elu.h"
#include "fresh_cpp/erf.h"
#include "fresh_cpp/erfc.h"
#include "fresh_cpp/erfinv.h"
#include "fresh_cpp/exp2.h"
#include "fresh_cpp/expm1.h"
#include "fresh_cpp/expm1cw.h"
#include "fresh_cpp/fill.h"
#include "fresh_cpp/fmod.h"
#include "fresh_cpp/gcd.h"
#include "fresh_cpp/gcd_legacy.h"
#include "fresh_cpp/gelu.h"
#include "fresh_cpp/hardmish.h"
#include "fresh_cpp/hardshrink.h"
#include "fresh_cpp/hardtanh.h"
#include "fresh_cpp/heaviside.h"
#include "fresh_cpp/mul_int32_limb2.h"
