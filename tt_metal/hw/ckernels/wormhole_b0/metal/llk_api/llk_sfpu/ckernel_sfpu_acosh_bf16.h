// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_sfpu_acosh_bf16_core_bridge_rational.h"
#pragma push_macro("USE_BF16")
#undef USE_BF16
#pragma push_macro("TT_SELECTED_CORE_TOTAL_FORM")
#undef TT_SELECTED_CORE_TOTAL_FORM
#pragma push_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#undef TT_SELECTED_CORE_TOTAL_SINGLE_ROW
#pragma push_macro("TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1")
#undef TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1
namespace ckernel::sfpu::AcoshBf16Config_source {
using namespace sfpi;
using namespace ::ckernel::sfpu::ttpoly::acosh;
constexpr uint32_t NUM_DEGREE = 4, DEN_DEGREE = 4;
constexpr uint32_t NUM_SEGMENTS = 1, LUT_SIZE = 12;
constexpr std::array<float, LUT_SIZE> LUT_DATA = {
    {1.0100000000000000e+00f,
     1.0000000000000000e+02f,
     -3.7053912878036499e-01f,
     1.0164581239223480e-01f,
     1.0282840728759766e+00f,
     -7.3719167709350586e-01f,
     -2.2856950759887695e-02f,
     -4.1008967161178589e-01f,
     1.0000000000000000e+00f,
     -3.7154752016067505e-01f,
     -2.2631739079952240e-01f,
     -3.3971322700381279e-03f}};
#define USE_BF16 1

// anonymous selected-CSV total form; mature LUT payload is unchanged
#define TT_SELECTED_CORE_TOTAL_FORM 1
#define TT_SELECTED_CORE_TOTAL_SINGLE_ROW 1
constexpr float TT_SELECTED_CORE_LOWER = 1.0156250000000000e+00f;
constexpr float TT_SELECTED_CORE_UPPER = 1.0000000000000000e+02f;
#define TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1 1
constexpr float TT_SELECTED_CORE_INVALID_BELOW = 1.0000000000000000e+00f;
constexpr float TT_SELECTED_CORE_ENDPOINT = 1.0000000000000000e+00f;
constexpr float TT_SELECTED_CORE_BRIDGE_UPPER = 1.0156250000000000e+00f;
constexpr float TT_SELECTED_CORE_BRIDGE_SLOPE = 1.6000000000000000e+01f;
constexpr int TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT = 132;
constexpr float TT_SELECTED_CORE_LOG_TWO = 6.9314718055994529e-01f;

#include "ckernel_sfpu_acosh_bf16_core_bridge_exponent.inc"
#include "ckernel_sfpu_acosh_bf16_rational_interleaved_core.inc"
template <uint32_t N, uint32_t D>
__attribute__((always_inline)) inline void eval_rational_numer_denom(
    const float* n, const float* d, vFloat x, vFloat& a, vFloat& b) {
    eval_rational_interleaved_numer_denom<N, D>(n, d, x, a, b);
}
#include "ckernel_sfpu_acosh_bf16_rational_segment_core.inc"
__attribute__((always_inline)) inline vFloat prepare_raw_domain_input(vFloat x) {
    return selected_core_bridge_prepare(x);
}
__attribute__((always_inline)) inline vFloat apply_output_postcompose(vFloat y, vFloat) { return y; }
__attribute__((always_inline)) inline void finalize_raw_domain_actions(vFloat x, vFloat& y) {
    selected_core_bridge_finalize(x, y);
}
// Upstream LLK runs several tiles in one DEST half, so the coefficients are
// immediates rather than rows parked past this tile.
inline void tile() {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1, COEFF_OFFSET = NUM_SEGMENTS + 1;
    const auto& lut = LUT_DATA;
#include "ckernel_sfpu_acosh_bf16_rational_scalar_tile.inc"
}
}  // namespace ckernel::sfpu::AcoshBf16Config_source
namespace ckernel::sfpu {
struct AcoshBf16Config {
    static inline void tile() { AcoshBf16Config_source::tile(); }
};
}  // namespace ckernel::sfpu
#pragma pop_macro("TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_FORM")
#pragma pop_macro("USE_BF16")

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_acosh() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    const std::uint32_t format = ckernel::math::src_zero_flag_srcb_fmt;
    return format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 32>
inline void calculate_acosh_bf16_body() {
    ckernel::sfpu::bf16::acosh::calculate_core_bridge_rational<ckernel::sfpu::AcoshBf16Config, ITERATIONS>();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_acosh_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3ea57ebbu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0x3fba2e90u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x4007c1f2u);
    calculate_acosh_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x33b17218u);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0x3eba2000u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0xbf004000u);
}
// The call sets the constants it holds and puts stock's back, so the init leaves them to stock's.
inline void init_acosh_bf16() {}

}  // namespace ckernel::sfpu
