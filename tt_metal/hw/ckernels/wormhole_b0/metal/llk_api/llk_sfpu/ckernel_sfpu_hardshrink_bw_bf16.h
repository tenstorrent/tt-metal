// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <array>
#include <cstdint>
#include <limits>
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_horner.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_dense_polynomial.h"
namespace sfpi {
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_mirrored_terminals.h"
}
#if !defined(ARCH_WORMHOLE)
#error "selected factor target differs from its compiled schedule"
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) ||                               \
    defined(ASYMPTOTIC_FACTOR_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_X) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || \
    defined(BASIS_AFFINE_EVEN) || defined(BASIS_CLAMP_MAX) || defined(BASIS_INPUT_ABS_X) ||                            \
    defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_MUL_ABS_X_BEFORE_POST) || defined(BASIS_MUL_SQRT_1_MINUS_ABS) ||    \
    defined(BASIS_POST_REFLECT_PI) || defined(BASIS_POST_SIGN_X) || defined(BASIS_RIGHT_TAIL_ABS_AFFINE) ||            \
    defined(BASIS_RIGHT_TAIL_IDENTITY) || defined(DISABLE_ADAPTIVE_DEGREE) || defined(DST_COEFF_DISABLE) ||            \
    defined(DST_COEFF_PROBE) || defined(DUAL_EVAL_DISABLE) || defined(HAS_CRITICAL_POINT) ||                           \
    defined(POLY_PARITY_EVEN) || defined(POLY_PARITY_ODD) || defined(POLY_TTI_DISABLE) ||                              \
    defined(RANGE_REDUCTION_CBRT) || defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_LOG) ||                   \
    defined(RANGE_REDUCTION_TAN) || defined(RANGE_REDUCTION_TRIG) || defined(TT_ACT_EVAL_POLY_CASCADE) ||              \
    defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS) ||                                                    \
    defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP) || defined(TT_DOMAIN_ACTION_PROGRAM) ||                         \
    defined(TT_DOMAIN_ACTION_RAW_TERMINAL_ENVELOPE) || defined(TT_DOMAIN_ACTION_TERMINAL_INGRESS_ONLY) ||              \
    defined(TT_DOMAIN_ACTION_WH_ORDERED_INGRESS) || defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) ||           \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) ||                                                     \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) ||           \
    defined(TT_INLINE_PROGRAM_RAW_ZERO_BOUNDARY) || defined(TT_INLINE_PROGRAM_RESULT_TERMINAL_RAW_OVERRIDE) ||         \
    defined(TT_LINEAR_BF16_UNDERFLOW_COUNT) || defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND) ||               \
    defined(TT_SELECTED_CORE_AGGREGATE_STIRLING) || defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1) ||         \
    defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1) ||                                                             \
    defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1) || defined(TT_SELECTED_CORE_TOTAL_COMPACT_LOOP_ROW) ||   \
    defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD) || defined(TT_SELECTED_CORE_TOTAL_FORM) ||                    \
    defined(TT_SELECTED_CORE_TOTAL_INPLACE_FACTORED_OVERLAY_V1) ||                                                     \
    defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) || defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW) ||            \
    defined(TT_SELECTED_WH_AGGREGATE_STIRLING) || defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL) ||                     \
    defined(TT_SELECTED_ZONE_GRADIENT) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) ||                             \
    defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO) || defined(TT_SPECIAL_COMPARE_NEG_INF) ||                           \
    defined(TT_SPECIAL_COMPARE_POS_INF) || defined(TT_SPECIAL_COMPARE_POS_ZERO) ||                                     \
    defined(TT_TARGET_BH_BF16_ACTION_COORDINATE) || defined(TT_TARGET_BH_BF16_PACK_RELU_RAW_NEG_NAN_REPAIR) ||         \
    defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR) || defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) ||      \
    defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT) ||           \
    defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT) ||           \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR) ||                                                      \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) ||    \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR) ||        \
    defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) ||    \
    defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT) || defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR) ||     \
    defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) ||                                                      \
    defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) ||    \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT) || defined(TT_TARGET_BH_BF16_SPECIAL_FINALIZER) ||           \
    defined(TT_TARGET_BH_FP32_RAW_NEG_ZERO_DISCRIMINATOR) || defined(TT_WH_CORE_SAME_ROW_GRAD_FINALIZE) ||             \
    defined(TT_WH_DST_CERTIFIED_MIRROR) || defined(TT_WH_DST_COEFF_ROW_UNROLL) ||                                      \
    defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS) || defined(TT_WH_FACTORED_CORE_MIRROR_FOLD) ||                          \
    defined(TT_WH_NORMALIZED_LOG1P_TERMINALS) || defined(TT_WH_RAW_NAN_UNION)
#error "mask factor cannot inherit alternate numerical selectors"
#endif
#pragma push_macro("DST_COEFF_STORE")
#undef DST_COEFF_STORE
#pragma push_macro("EMBEDDED_LUT")
#undef EMBEDDED_LUT
#pragma push_macro("EVAL_METHOD_POLY_CASCADE")
#undef EVAL_METHOD_POLY_CASCADE
#pragma push_macro("FUSE_GRAD_MUL")
#undef FUSE_GRAD_MUL
#pragma push_macro("HAS_SEGMENT_DEGREES")
#undef HAS_SEGMENT_DEGREES
#pragma push_macro("SEG0_DEGREE")
#undef SEG0_DEGREE
#pragma push_macro("SEG1_DEGREE")
#undef SEG1_DEGREE
#pragma push_macro("SEG2_DEGREE")
#undef SEG2_DEGREE
#pragma push_macro("TT_ACT_EVAL_KIND")
#undef TT_ACT_EVAL_KIND
#pragma push_macro("TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT")
#undef TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT
#pragma push_macro("TT_SPECIAL_NAN")
#undef TT_SPECIAL_NAN
#pragma push_macro("TT_SPECIAL_NEG_INF")
#undef TT_SPECIAL_NEG_INF
#pragma push_macro("TT_SPECIAL_NEG_ZERO")
#undef TT_SPECIAL_NEG_ZERO
#pragma push_macro("TT_SPECIAL_POS_INF")
#undef TT_SPECIAL_POS_INF
#pragma push_macro("TT_SPECIAL_POS_ZERO")
#undef TT_SPECIAL_POS_ZERO
#pragma push_macro("TT_SPECIAL_VALUE_POLICY")
#undef TT_SPECIAL_VALUE_POLICY
#pragma push_macro("TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL")
#undef TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL
#pragma push_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#undef TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT
#pragma push_macro("TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED")
#undef TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED
#pragma push_macro("USE_BF16")
#undef USE_BF16
#pragma push_macro("USE_DUAL_EVAL")
#undef USE_DUAL_EVAL
#define FUSE_GRAD_MUL 1
#define USE_BF16 1
#define USE_DUAL_EVAL 1
namespace ttpoly_generated::HardshrinkBwBf16Config_detail {
using namespace ::sfpi;
using ::sfpi::DataLayout;
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated by run_csv.sh
// Degree 1, 3 segments, range [-10.0, 10.0]

#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = 1;
constexpr uint32_t NUM_SEGMENTS = 3;

constexpr float INPUT_MIN = -1.0000000000000000e+01f;
constexpr float INPUT_MAX = 1.0000000000000000e+01f;

constexpr uint32_t LUT_SIZE_BF16 = 10;
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {
    {-1.0000000000000000e+01f,
     -5.0000000000000000e-01f,
     5.0000005960000005e-01f,
     1.0000000000000000e+01f,
     1.0000000000000000e+00f,
     0.0000000000000000e+00f,
     0.0000000000000000e+00f,
     0.0000000000000000e+00f,
     1.0000000000000000e+00f,
     0.0000000000000000e+00f}};

constexpr uint32_t LUT_SIZE_FP32 = 10;
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {
    {-1.0000000000000000e+01f,
     -5.0000000000000000e-01f,
     5.0000005960000005e-01f,
     1.0000000000000000e+01f,
     1.0000000000000000e+00f,
     0.0000000000000000e+00f,
     0.0000000000000000e+00f,
     0.0000000000000000e+00f,
     1.0000000000000000e+00f,
     0.0000000000000000e+00f}};

#ifdef USE_BF16
// static: a namespace-scope constexpr REFERENCE has EXTERNAL linkage (the
// internal-linkage-for-const rule covers const objects, not references),
// so without static the reference symbol pins the whole LUT array into the
// local-data-memory image of every TRISC (~1.7KB free) -- s256+ configs
// then fail to link on any kernel that reads the LUT at runtime (segment
// overflows region:1). Internal linkage lets LTO discard the dead copy;
// compile-time consumers are unaffected.
static constexpr auto& LUT_DATA = LUT_DATA_BF16;
constexpr uint32_t LUT_SIZE = LUT_SIZE_BF16;
#else
static constexpr auto& LUT_DATA = LUT_DATA_FP32;
constexpr uint32_t LUT_SIZE = LUT_SIZE_FP32;
#endif

// eval_method: poly_cascade (default piecewise polynomial cascade)
#define TT_ACT_EVAL_KIND TT_ACT_EVAL_POLY_CASCADE
#define EVAL_METHOD_POLY_CASCADE
#define DST_COEFF_STORE

#ifndef DISABLE_ADAPTIVE_DEGREE
#define SEG0_DEGREE 0
#define SEG1_DEGREE 0
#define SEG2_DEGREE 0
#define HAS_SEGMENT_DEGREES
constexpr uint32_t SEGMENT_DEGREES[] = {0, 0, 0};
#endif

// Declared special-value policy from the typed activation specification.
// 0=NaN 1=+Inf 2=-Inf 3=+0 4=-0 5=finite_other(pass through)
#define TT_SPECIAL_VALUE_POLICY
#define TT_SPECIAL_NAN 5
#define TT_SPECIAL_POS_INF 5
#define TT_SPECIAL_NEG_INF 5
#define TT_SPECIAL_POS_ZERO 3
#define TT_SPECIAL_NEG_ZERO 3
#if !defined(FUSE_GRAD_MUL) || !defined(USE_BF16)
#error "certified gradient selection requires the BF16 two-input ABI"
#endif
#if defined(DISABLE_ADAPTIVE_DEGREE) || defined(DST_COEFF_PROBE) || defined(TT_SELECTED_CORE_TOTAL_COMPACT_LOOP_ROW)
#error "certified gradient selection requires the selected constant cascade"
#endif
static_assert(SEGMENT_DEGREES[0] == 0, "certified constant degree changed");
static_assert(SEGMENT_DEGREES[1] == 0, "certified constant degree changed");
static_assert(SEGMENT_DEGREES[2] == 0, "certified constant degree changed");
// Selected zero-mask identity: e125ed321915f8fd899b082c5442dd1b03e4b5f60d59baeb4737945cfe5718de
#define TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT 1

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_prepare.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_action_coordinate.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_raw_class_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_target_special_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_encoded_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_dual_tile.inc"
inline void tile() { piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(LUT_DATA); }

}  // namespace ttpoly_generated::HardshrinkBwBf16Config_detail
namespace ttpoly_generated {
struct HardshrinkBwBf16Config {
    static inline void tile() { HardshrinkBwBf16Config_detail::tile(); }
};
}  // namespace ttpoly_generated
#pragma pop_macro("USE_DUAL_EVAL")
#pragma pop_macro("USE_BF16")
#pragma pop_macro("TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED")
#pragma pop_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#pragma pop_macro("TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL")
#pragma pop_macro("TT_SPECIAL_VALUE_POLICY")
#pragma pop_macro("TT_SPECIAL_POS_ZERO")
#pragma pop_macro("TT_SPECIAL_POS_INF")
#pragma pop_macro("TT_SPECIAL_NEG_ZERO")
#pragma pop_macro("TT_SPECIAL_NEG_INF")
#pragma pop_macro("TT_SPECIAL_NAN")
#pragma pop_macro("TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT")
#pragma pop_macro("TT_ACT_EVAL_KIND")
#pragma pop_macro("SEG2_DEGREE")
#pragma pop_macro("SEG1_DEGREE")
#pragma pop_macro("SEG0_DEGREE")
#pragma pop_macro("HAS_SEGMENT_DEGREES")
#pragma pop_macro("FUSE_GRAD_MUL")
#pragma pop_macro("EVAL_METHOD_POLY_CASCADE")
#pragma pop_macro("EMBEDDED_LUT")
#pragma pop_macro("DST_COEFF_STORE")
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_config_tile.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_hardshrink_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::HardshrinkBwBf16Config, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
