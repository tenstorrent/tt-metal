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
    for (int row = 0; row < ITERATIONS; ++row) {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_exp_21f_bf16_<false>(input);
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

}  // namespace ckernel::sfpu
