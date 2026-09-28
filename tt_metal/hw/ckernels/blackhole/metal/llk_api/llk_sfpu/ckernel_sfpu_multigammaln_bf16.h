// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_core_bridge_rational.h"
namespace sfpi {
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_mirrored_terminals.h"
}
#if defined(FUSE_GRAD_MUL) || defined(PRECOMPOSE_INPUT_AFFINE) || defined(RANGE_REDUCTION_EXP) ||             \
    defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_TRIG) || defined(RATIONAL_DEN_PARITY_EVEN) ||     \
    defined(RATIONAL_NUM_PARITY_ODD) || defined(RATIONAL_RECIPROCAL_ONE_ITER) ||                              \
    defined(RATIONAL_RECIPROCAL_ZERO_ITER) || defined(TT_BF16_RAW_UINT16_OUTPUT) ||                           \
    defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP) || defined(TT_DOMAIN_ACTION_PROGRAM) ||                \
    defined(TT_INLINE_PROGRAM_RATIONAL_LATE_RAW_SCHEDULE) ||                                                  \
    defined(TT_MIRRORED_RATIONAL_TYPED_RAW_NEG_NAN_TERMINAL) || defined(TT_RATIONAL_COORDINATE_BOUND) ||      \
    defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND) ||                                                 \
    defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1) || defined(TT_SELECTED_CORE_TOTAL_FORM) ||        \
    defined(TT_SELECTED_WH_AGGREGATE_STIRLING) || defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL) ||            \
    defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO) ||        \
    defined(TT_SPECIAL_COMPARE_POS_INF) || defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR) ||          \
    defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT) ||      \
    defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT) ||  \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT) || defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT) || defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) ||   \
    defined(TT_TARGET_BH_FP32_RAW_NEG_ZERO_DISCRIMINATOR) || defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS) ||    \
    defined(TT_WH_NORMALIZED_LOG1P_TERMINALS) || defined(TT_WH_PARITY_NUMERATOR_PIN)
#error "aggregate cannot inherit alternate numerical selectors"
#endif
#pragma push_macro("USE_BF16")
#undef USE_BF16
#pragma push_macro("TT_SELECTED_CORE_AGGREGATE_STIRLING")
#undef TT_SELECTED_CORE_AGGREGATE_STIRLING
#pragma push_macro("TT_SELECTED_AGGREGATE_PUBLIC_LGAMMA_CLASS_REPAIR")
#undef TT_SELECTED_AGGREGATE_PUBLIC_LGAMMA_CLASS_REPAIR
#pragma push_macro("TT_SPECIAL_VALUE_POLICY")
#undef TT_SPECIAL_VALUE_POLICY
#pragma push_macro("TT_TARGET_BH_BF16_ACTION_COORDINATE")
#undef TT_TARGET_BH_BF16_ACTION_COORDINATE
#pragma push_macro("TT_TARGET_BH_BF16_SPECIAL_FINALIZER")
#undef TT_TARGET_BH_BF16_SPECIAL_FINALIZER
#pragma push_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#undef TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT
#pragma push_macro("TT_SPECIAL_COMPARE_NEG_INF")
#undef TT_SPECIAL_COMPARE_NEG_INF
#pragma push_macro("TT_SPECIAL_COMPARE_POS_ZERO")
#undef TT_SPECIAL_COMPARE_POS_ZERO
#pragma push_macro("TT_SPECIAL_NAN")
#undef TT_SPECIAL_NAN
#pragma push_macro("TT_SPECIAL_POS_INF")
#undef TT_SPECIAL_POS_INF
#pragma push_macro("TT_SPECIAL_NEG_INF")
#undef TT_SPECIAL_NEG_INF
#pragma push_macro("TT_SPECIAL_POS_ZERO")
#undef TT_SPECIAL_POS_ZERO
#pragma push_macro("TT_SPECIAL_NEG_ZERO")
#undef TT_SPECIAL_NEG_ZERO
#pragma push_macro("TT_RATIONAL_DST_COEFF")
#undef TT_RATIONAL_DST_COEFF
namespace ttpoly_generated::MultigammalnBf16Config_source {
using namespace sfpi;
constexpr uint32_t NUM_DEGREE = 6, DEN_DEGREE = 4;
constexpr uint32_t NUM_SEGMENTS = 1, LUT_SIZE = 14;
constexpr std::array<float, LUT_SIZE> LUT_DATA = {
    {1.5009765625000000e+00f,
     5.0000000000000000e+01f,
     6.4772596359252930e+00f,
     -9.3999099731445312e+00f,
     2.1342811584472656e+00f,
     2.6674137115478516e+00f,
     -1.6923900842666626e+00f,
     3.1229573488235474e-01f,
     1.2563235359266400e-03f,
     -2.9412943124771118e-01f,
     1.0000000000000000e+00f,
     -8.9121544361114502e-01f,
     1.9727556407451630e-01f,
     2.6383467018604279e-02f}};
#define USE_BF16 1

// Certified selected rational core plus aggregate Stirling tail.
#define TT_SELECTED_CORE_AGGREGATE_STIRLING 1
constexpr uint32_t TT_SELECTED_STIRLING_CORE_LOWER_BITS = 0x3fc00000u;
constexpr uint32_t TT_SELECTED_STIRLING_CORE_UPPER_BITS = 0x42480000u;
constexpr uint32_t TT_SELECTED_STIRLING_OVERFLOW_BITS = 0x7b47f955u;
constexpr uint32_t TT_SELECTED_STIRLING_AGGREGATE_COUNT = 4u;
constexpr uint32_t TT_SELECTED_STIRLING_LOG_OFFSET_BITS = 0x40a00000u;
constexpr uint32_t TT_SELECTED_STIRLING_CONSTANT_BITS = 0x40e384a9u;
constexpr uint32_t TT_SELECTED_STIRLING_CORRECTION_SCALE_BITS = 0x41000000u;
constexpr std::array<float, 11> TT_SELECTED_STIRLING_LOG_RATIO = {
    {1.0000000000000000e+00f,
     -4.9999997019767761e-01f,
     3.3333346247673035e-01f,
     -2.5000315904617310e-01f,
     1.9999562203884125e-01f,
     -1.6651614010334015e-01f,
     1.4271874725818634e-01f,
     -1.2755781412124634e-01f,
     1.1764620989561081e-01f,
     -9.1233491897583008e-02f,
     3.7215668708086014e-02f}};
constexpr std::array<float, 13> TT_SELECTED_STIRLING_CORRECTION = {
    {2.7755575615628914e-17f,
     4.4791665673255920e-01f,
     2.9296875000000000e-02f,
     3.0097113922238350e-03f,
     3.7765502929687500e-04f,
     5.2835079259239137e-05f,
     7.8976299846544862e-06f,
     1.2328968068686663e-06f,
     1.9902516612546606e-07f,
     3.1603722305817428e-08f,
     6.6526548714307410e-09f,
     1.2773651580921808e-10f,
     4.3693662576949066e-10f}};
#define TT_SELECTED_AGGREGATE_PUBLIC_LGAMMA_CLASS_REPAIR 1
constexpr uint32_t TT_SELECTED_AGGREGATE_CLASS_TERM_COUNT = 2u;
constexpr std::array<uint32_t, 2> TT_SELECTED_AGGREGATE_CLASS_SHIFT_BITS = {{0x3f800000u, 0x3fc00000u}};
constexpr uint32_t TT_SELECTED_AGGREGATE_BINARY_POLE_SHIFT_BITS = 0x3f800000u;
constexpr std::array<uint32_t, 2> TT_SELECTED_AGGREGATE_BINARY_POLE_SFPU_SCOPE_BITS = {{0xbb800000u, 0x3b800000u}};
constexpr std::array<uint32_t, 2> TT_SELECTED_AGGREGATE_BINARY_POLE_TARGET_OPEN_BITS = {{0xbb600000u, 0x3b200000u}};
constexpr uint32_t TT_SELECTED_AGGREGATE_ZERO_REPRESENTATIVE_COUNT = 1u;
constexpr std::array<uint32_t, 1> TT_SELECTED_AGGREGATE_ZERO_REPRESENTATIVE_BITS = {{0xc02b0000u}};

// Declared special-value policy from the typed activation specification.
// 0=NaN 1=+Inf 2=-Inf 3=+0 4=-0 5=finite_other(pass through)
#define TT_SPECIAL_VALUE_POLICY
#define TT_TARGET_BH_BF16_ACTION_COORDINATE
#define TT_TARGET_BH_BF16_SPECIAL_FINALIZER
#define TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT 2
#define TT_SPECIAL_COMPARE_NEG_INF
#define TT_SPECIAL_COMPARE_POS_ZERO
#define TT_SPECIAL_NAN 0
#define TT_SPECIAL_POS_INF 1
#define TT_SPECIAL_NEG_INF 0
#define TT_SPECIAL_POS_ZERO 0
#define TT_SPECIAL_NEG_ZERO 0

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_aggregate_stirling_core.inc"
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
inline vFloat prepare_raw_domain_input(vFloat x) { return selected_aggregate_stirling_prepare(x); }
inline vFloat apply_output_postcompose(vFloat y, vFloat) { return y; }
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_target_special_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_action_coordinate.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_finalize.inc"
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
}  // namespace ttpoly_generated::MultigammalnBf16Config_source
namespace ttpoly_generated {
struct MultigammalnBf16Config {
    static inline void tile() { MultigammalnBf16Config_source::tile(); }
};
}  // namespace ttpoly_generated
#pragma pop_macro("TT_RATIONAL_DST_COEFF")
#pragma pop_macro("TT_SPECIAL_NEG_ZERO")
#pragma pop_macro("TT_SPECIAL_POS_ZERO")
#pragma pop_macro("TT_SPECIAL_NEG_INF")
#pragma pop_macro("TT_SPECIAL_POS_INF")
#pragma pop_macro("TT_SPECIAL_NAN")
#pragma pop_macro("TT_SPECIAL_COMPARE_POS_ZERO")
#pragma pop_macro("TT_SPECIAL_COMPARE_NEG_INF")
#pragma pop_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#pragma pop_macro("TT_TARGET_BH_BF16_SPECIAL_FINALIZER")
#pragma pop_macro("TT_TARGET_BH_BF16_ACTION_COORDINATE")
#pragma pop_macro("TT_SPECIAL_VALUE_POLICY")
#pragma pop_macro("TT_SELECTED_AGGREGATE_PUBLIC_LGAMMA_CLASS_REPAIR")
#pragma pop_macro("TT_SELECTED_CORE_AGGREGATE_STIRLING")
#pragma pop_macro("USE_BF16")
#define TT_POLY_TT_POLY_AGGREGATE_MULTIGAMMALN_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_tt_poly_aggregate_multigammaln_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_core_bridge_rational<ttpoly_generated::MultigammalnBf16Config, ITERATIONS>();
}
inline void init_tt_poly_aggregate_multigammaln_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_core_bridge_rational<ttpoly_generated::MultigammalnBf16Config>();
}
#endif

}  // namespace ckernel::sfpu
