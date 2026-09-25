// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_core_bridge_rational.h"
#pragma push_macro("USE_BF16")
#undef USE_BF16
#pragma push_macro("TT_SELECTED_CORE_TOTAL_FORM")
#undef TT_SELECTED_CORE_TOTAL_FORM
#pragma push_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#undef TT_SELECTED_CORE_TOTAL_SINGLE_ROW
#pragma push_macro("TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1")
#undef TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1
#pragma push_macro("TT_RATIONAL_DST_COEFF")
#undef TT_RATIONAL_DST_COEFF
namespace ttpoly_generated::AcoshBf16Config_source {
using namespace sfpi;
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

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_core_bridge_exponent.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_rational_interleaved_core.inc"
template <uint32_t N, uint32_t D>
inline void eval_rational_numer_denom(const float* n, const float* d, vFloat x, vFloat& a, vFloat& b) {
    eval_rational_interleaved_numer_denom<N, D>(n, d, x, a, b);
}
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_rational_segment_core.inc"
#if defined(ARCH_BLACKHOLE)
#define TT_RATIONAL_DST_COEFF 1
#else
#define TT_RATIONAL_DST_COEFF 0
#endif
inline vFloat prepare_raw_domain_input(vFloat x) { return selected_core_bridge_prepare(x); }
inline vFloat apply_output_postcompose(vFloat y, vFloat) { return y; }
inline void finalize_raw_domain_actions(vFloat x, vFloat& y) { selected_core_bridge_finalize(x, y); }
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_rational_dst_core.inc"
inline void tile() {
#if defined(ARCH_BLACKHOLE)
    piecewise_rational_dst_coeff_tile(true);
#else
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1, COEFF_OFFSET = NUM_SEGMENTS + 1;
    const auto& lut = LUT_DATA;
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_rational_scalar_tile.inc"
#endif
}
}  // namespace ttpoly_generated::AcoshBf16Config_source
namespace ttpoly_generated {
struct AcoshBf16Config {
    static inline void tile() { AcoshBf16Config_source::tile(); }
};
}  // namespace ttpoly_generated
#pragma pop_macro("TT_RATIONAL_DST_COEFF")
#pragma pop_macro("TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_FORM")
#pragma pop_macro("USE_BF16")
