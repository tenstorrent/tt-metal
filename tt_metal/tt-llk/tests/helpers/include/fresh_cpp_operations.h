// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only semantic implementations used by the corpus A/Bs.  These deliberately
// state the math one row at a time: no fixed LREGs, raw instructions, replay slots,
// SFPLOADMACRO templates, or hand-interleaved software schedule.
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

template <int ITERATIONS>
inline void calculate_sigmoid_appx_pwl_fresh_cpp() {
    // The same piecewise-linear function the production kernel computes,
    // stated as ordinary typed C++: six disjoint |x| ranges select an
    // (A, B) pair, one fused multiply-add, sign restore, +0.5.  Every
    // coefficient below is exactly representable in the fp16 LUT entry
    // format (they are the decoded production table values); the range
    // breakpoints are {0.5, 1, 1.5, 2, 4}.
    for (int row = 0; row < ITERATIONS; ++row) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat ax = sfpi::abs(x);
        sfpi::vFloat a = 0.0f;
        sfpi::vFloat b = 0.499755859375f;
        v_if (ax < 0.5f)                { a = 0.2452392578125f;   b = -0.000499725341796875f; } v_endif;
        v_if (ax >= 0.5f && ax < 1.0f)  { a = 0.21728515625f;     b = 0.01519775390625f; }      v_endif;
        v_if (ax >= 1.0f && ax < 1.5f)  { a = 0.173095703125f;    b = 0.05987548828125f; }      v_endif;
        v_if (ax >= 1.5f && ax < 2.0f)  { a = 0.126220703125f;    b = 0.1297607421875f; }       v_endif;
        v_if (ax >= 2.0f && ax < 4.0f)  { a = 0.048492431640625f; b = 0.2998046875f; }          v_endif;
        sfpi::vFloat r = a * ax + b;
        r = sfpi::copysgn(r, x);
        sfpi::dst_reg[0] = r + 0.5f;
        sfpi::dst_reg++;
    }
}

template <bool IS_MAX, int ITERATIONS>
inline void calculate_binary_max_min_fresh_cpp(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out) {
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 8
    for (int row = 0; row < ITERATIONS; ++row) {
        const sfpi::vFloat lhs = sfpi::dst_reg[dst_index_in0 * tile_rows];
        const sfpi::vFloat rhs = sfpi::dst_reg[dst_index_in1 * tile_rows];
        sfpi::dst_reg[dst_index_out * tile_rows] = IS_MAX ? sfpi::max(lhs, rhs) : sfpi::min(lhs, rhs);
        sfpi::dst_reg++;
    }
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
