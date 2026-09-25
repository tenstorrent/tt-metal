// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <array>
#include <cstdint>
#include <limits>
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_horner.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_dense_polynomial.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
namespace sfpi {
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_min_max.h"
}
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
    defined(BASIS_RIGHT_TAIL_IDENTITY) || defined(DENSE_MIRROR_FOLD_DISABLE) || defined(DISABLE_ADAPTIVE_DEGREE) ||    \
    defined(DST_COEFF_DISABLE) || defined(DST_COEFF_PROBE) || defined(DUAL_EVAL_DISABLE) ||                            \
    defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1) || defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) ||      \
    defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) || defined(EVAL_METHOD_TAN_STANDALONE) ||            \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) || defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) ||                   \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || defined(EXP_HW_COMPOSE_SIGMOID) ||                                \
    defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) ||                    \
    defined(HAS_CRITICAL_POINT) || defined(POLY_PARITY_EVEN) || defined(POLY_PARITY_ODD) ||                            \
    defined(POLY_TTI_DISABLE) || defined(POW_HW_RECIPROCAL) || defined(RANGE_REDUCTION_CBRT) ||                        \
    defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_RECIP_COMPLEMENT) ||       \
    defined(RANGE_REDUCTION_TAN) || defined(RANGE_REDUCTION_TRIG) || defined(REDUCE_EXP_BASE2) ||                      \
    defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) ||                                  \
    defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD) || defined(REDUCE_EXP_COMPOSE_SELU) ||                                  \
    defined(REDUCE_EXP_COMPOSE_SIGMOID) || defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) ||                              \
    defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED) || defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT) ||                  \
    defined(REDUCE_RECIP_COMPLEMENT) || defined(TT_ABS_RESIDUAL_AFFINE_1_NEEDS_RECIPROCAL) ||                          \
    defined(TT_ACT_EVAL_POLY_CASCADE) || defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS) ||               \
    defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP) || defined(TT_DOMAIN_ACTION_PROGRAM) ||                         \
    defined(TT_DOMAIN_ACTION_RAW_TERMINAL_ENVELOPE) || defined(TT_DOMAIN_ACTION_TERMINAL_INGRESS_ONLY) ||              \
    defined(TT_DOMAIN_ACTION_WH_ORDERED_INGRESS) || defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) ||           \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) ||                                                     \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_INLINE_PROGRAM_GRADIENT_SCALE) ||            \
    defined(TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT) || defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) ||                  \
    defined(TT_INLINE_PROGRAM_NEEDS_RECIPROCAL) || defined(TT_INLINE_PROGRAM_RAW_ZERO_BOUNDARY) ||                     \
    defined(TT_INLINE_PROGRAM_RESULT_TERMINAL_RAW_OVERRIDE) ||                                                         \
    defined(TT_INLINE_PROGRAM_SELECTED_FACTOR_INTRINSIC_CLAMP) || defined(TT_LINEAR_BF16_UNDERFLOW_COUNT) ||           \
    defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND) || defined(TT_SELECTED_CORE_AGGREGATE_STIRLING) ||          \
    defined(TT_SELECTED_CORE_BRIDGE_DIRECT_LOG_TAIL_1) || defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1) ||   \
    defined(TT_SELECTED_CORE_EXP2_SHIFTED) || defined(TT_SELECTED_CORE_EXP_TAIL_ABOVE) ||                              \
    defined(TT_SELECTED_CORE_GAUSSIAN_MILLS_TAIL_1) || defined(TT_SELECTED_CORE_NEEDS_RECIPROCAL) ||                   \
    defined(TT_SELECTED_CORE_ONE_SIDED_CODY_EXP_TAIL_1) || defined(TT_SELECTED_CORE_SCALED_EXP_CORRECTION_TAIL_1) ||   \
    defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1) ||                                                             \
    defined(TT_SELECTED_CORE_SIGMOID_PRODUCT_DERIVATIVE_TAIL_1) ||                                                     \
    defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1) ||                                                       \
    defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1) || defined(TT_SELECTED_CORE_SYMMETRIC_EXP_TAIL_1) ||         \
    defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD) ||                                                            \
    defined(TT_SELECTED_CORE_TOTAL_INPLACE_FACTORED_OVERLAY_V1) || defined(TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL) || \
    defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) ||                                                          \
    defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) || defined(TT_SELECTED_WH_AGGREGATE_STIRLING) ||        \
    defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL) || defined(TT_SELECTED_ZONE_GRADIENT) ||                             \
    defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO) ||                 \
    defined(TT_SPECIAL_COMPARE_NEG_INF) || defined(TT_SPECIAL_COMPARE_POS_INF) ||                                      \
    defined(TT_SPECIAL_COMPARE_POS_ZERO) || defined(TT_SQUARE_DECAY_NEEDS_RECIPROCAL) ||                               \
    defined(TT_TARGET_BH_BF16_ACTION_COORDINATE) || defined(TT_TARGET_BH_BF16_PACK_RELU_RAW_NEG_NAN_REPAIR) ||         \
    defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR) || defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) ||      \
    defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT) ||           \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR) ||                                                      \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) ||    \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR) ||        \
    defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR) ||                                                            \
    defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR) ||                                                      \
    defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) ||                                                      \
    defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) ||    \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT) || defined(TT_TARGET_BH_BF16_SPECIAL_FINALIZER) ||           \
    defined(TT_TARGET_BH_FP32_RAW_NEG_ZERO_DISCRIMINATOR) || defined(TT_WH_DST_CERTIFIED_MIRROR) ||                    \
    defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS) || defined(TT_WH_FACTORED_CORE_MIRROR_FOLD) ||                          \
    defined(TT_WH_NORMALIZED_LOG1P_TERMINALS) || defined(TT_WH_RAW_NAN_UNION) ||                                       \
    defined(TT_WH_REDUCED_POLY_UNROLL32) || defined(TT_WH_SIGNED_ABS_COEFFICIENT_STORE) ||                             \
    defined(TT_WH_SIGNED_ABS_UNROLL32) || defined(TT_WH_SYMMETRIC_EXP_POOL)
#error "mask factor cannot inherit alternate numerical selectors"
#endif
#pragma push_macro("DST_COEFF_ELIGIBLE")
#undef DST_COEFF_ELIGIBLE
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
#pragma push_macro("SEG3_DEGREE")
#undef SEG3_DEGREE
#pragma push_macro("SEG4_DEGREE")
#undef SEG4_DEGREE
#pragma push_macro("SEG5_DEGREE")
#undef SEG5_DEGREE
#pragma push_macro("SEG6_DEGREE")
#undef SEG6_DEGREE
#pragma push_macro("SEG7_DEGREE")
#undef SEG7_DEGREE
#pragma push_macro("TT_ACT_EVAL_KIND")
#undef TT_ACT_EVAL_KIND
#pragma push_macro("TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1")
#undef TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1
#pragma push_macro("TT_SELECTED_CORE_TOTAL_FORM")
#undef TT_SELECTED_CORE_TOTAL_FORM
#pragma push_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#undef TT_SELECTED_CORE_TOTAL_SINGLE_ROW
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
#pragma push_macro("TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR")
#undef TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR
#pragma push_macro("TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT")
#undef TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT
#pragma push_macro("TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR")
#undef TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR
#pragma push_macro("TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT")
#undef TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT
#pragma push_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#undef TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT
#pragma push_macro("TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED")
#undef TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED
#pragma push_macro("TT_WH_CORE_SAME_ROW_GRAD_FINALIZE")
#undef TT_WH_CORE_SAME_ROW_GRAD_FINALIZE
#pragma push_macro("TT_WH_DENSE_DST_COEFF_STORE")
#undef TT_WH_DENSE_DST_COEFF_STORE
#pragma push_macro("TT_WH_DST_COEFF_ROW_UNROLL")
#undef TT_WH_DST_COEFF_ROW_UNROLL
#pragma push_macro("USE_BF16")
#undef USE_BF16
#pragma push_macro("USE_DUAL_EVAL")
#undef USE_DUAL_EVAL
#define FUSE_GRAD_MUL 1
#define USE_BF16 1
#define USE_DUAL_EVAL 1
namespace ttpoly_generated::CeluBwBf16Config_detail {
using namespace ::sfpi;
using ::sfpi::DataLayout;
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated by run_csv.sh
// Degree 4, 8 segments, range [-10.0, 10.0]

#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = 4;
constexpr uint32_t NUM_SEGMENTS = 8;

constexpr float INPUT_MIN = -1.0000000000000000e+01f;
constexpr float INPUT_MAX = 1.0000000000000000e+01f;

constexpr uint32_t LUT_SIZE_BF16 = 49;
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {
    {-1.0000000000000000e+01f, -7.5000000000000000e+00f, -5.0000000000000000e+00f, -2.5000000000000000e+00f,
     0.0000000000000000e+00f,  2.5000000000000000e+00f,  5.0000000000000000e+00f,  7.5000000000000000e+00f,
     1.0000000000000000e+01f,  5.9430828782565615e-02f,  2.3029023816789652e-02f,  3.3860369916065961e-03f,
     2.2343112732202267e-04f,  5.5732246855855877e-06f,  2.4915979803064092e-01f,  1.2654657429635247e-01f,
     2.4672375621796437e-02f,  2.1777982812627890e-03f,  7.3143518001406987e-05f,  6.8450094809836959e-01f,
     4.8537219372974866e-01f,  1.3692817854458847e-01f,  1.7956171349555267e-02f,  9.1267888494335665e-04f,
     9.9849183161647559e-01f,  9.7942633387414701e-01f,  4.4502864939263725e-01f,  1.0789154754090804e-01f,
     1.1178449966878211e-02f,  1.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,
     0.0000000000000000e+00f,  0.0000000000000000e+00f,  1.0000000000000000e+00f,  0.0000000000000000e+00f,
     0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,  1.0000000000000000e+00f,
     0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,
     1.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,
     0.0000000000000000e+00f}};

constexpr uint32_t LUT_SIZE_FP32 = 49;
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {
    {-1.0000000000000000e+01f, -7.5000000000000000e+00f, -5.0000000000000000e+00f, -2.5000000000000000e+00f,
     0.0000000000000000e+00f,  2.5000000000000000e+00f,  5.0000000000000000e+00f,  7.5000000000000000e+00f,
     1.0000000000000000e+01f,  5.9430828782565615e-02f,  2.3029023816789652e-02f,  3.3860369916065961e-03f,
     2.2343112732202267e-04f,  5.5732246855855877e-06f,  2.4915979803064092e-01f,  1.2654657429635247e-01f,
     2.4672375621796437e-02f,  2.1777982812627890e-03f,  7.3143518001406987e-05f,  6.8450094809836959e-01f,
     4.8537219372974866e-01f,  1.3692817854458847e-01f,  1.7956171349555267e-02f,  9.1267888494335665e-04f,
     9.9849183161647559e-01f,  9.7942633387414701e-01f,  4.4502864939263725e-01f,  1.0789154754090804e-01f,
     1.1178449966878211e-02f,  1.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,
     0.0000000000000000e+00f,  0.0000000000000000e+00f,  1.0000000000000000e+00f,  0.0000000000000000e+00f,
     0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,  1.0000000000000000e+00f,
     0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,
     1.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,  0.0000000000000000e+00f,
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
#define TT_WH_DENSE_DST_COEFF_STORE 1
constexpr uint32_t TT_WH_DST_COEFF_BASE = 64u;
constexpr uint32_t TT_WH_DST_COEFF_LIMIT = 108u;
constexpr uint32_t TT_WH_DST_COEFF_COUNT = 20u;
constexpr bool TT_WH_DST_COEFF_SINGLE_ROW = true;
constexpr uint32_t TT_WH_DST_COEFF_MIRROR_SEGMENTS = 0u;
#define TT_WH_DST_COEFF_ROW_UNROLL 2

// anonymous selected-CSV total form; mature LUT payload is unchanged
#define TT_SELECTED_CORE_TOTAL_FORM 1
#define TT_SELECTED_CORE_TOTAL_SINGLE_ROW 1
constexpr float TT_SELECTED_CORE_LOWER = -1.0000000000000000e+01f;
constexpr float TT_SELECTED_CORE_UPPER = 1.0000000000000000e+01f;
#define TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1 1
constexpr float TT_SELECTED_CORE_NEGATIVE_TERMINAL = -8.7500000000000000e+01f;
constexpr float TT_SELECTED_CORE_POSITIVE_VALUE = 1.0000000000000000e+00f;
constexpr float TT_SELECTED_CORE_EXP2_MULT = 1.4426950408889634e+00f;
constexpr int TT_SELECTED_CORE_EXP2_BIAS = 127;
constexpr int TT_SELECTED_CORE_EXP2_OUTPUT_SHIFT = 0;
constexpr uint32_t TT_SELECTED_CORE_EXP_DEGREE = 3u;
constexpr float TT_SELECTED_CORE_EXP_COEFFS[] = {
    1.0000000000000000e+00f, 6.9556409120559692e-01f, 2.2616928815841675e-01f, 7.8141160309314728e-02f};

#ifndef DISABLE_ADAPTIVE_DEGREE
#define SEG0_DEGREE 4
#define SEG1_DEGREE 4
#define SEG2_DEGREE 4
#define SEG3_DEGREE 4
#define SEG4_DEGREE 0
#define SEG5_DEGREE 0
#define SEG6_DEGREE 0
#define SEG7_DEGREE 0
#define HAS_SEGMENT_DEGREES
constexpr uint32_t SEGMENT_DEGREES[] = {4, 4, 4, 4, 0, 0, 0, 0};
#endif
#define TT_WH_CORE_SAME_ROW_GRAD_FINALIZE 1

// Declared special-value policy from the typed activation specification.
// 0=NaN 1=+Inf 2=-Inf 3=+0 4=-0 5=finite_other(pass through)
#define TT_SPECIAL_VALUE_POLICY
#define TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR
#define TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT 3
constexpr float TT_TARGET_BH_BF16_RAW_NEG_NAN_CONSTANT = 0.0000000000000000e+00f;
#define TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR
#define TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT 1
constexpr float TT_TARGET_BH_BF16_RAW_POS_NAN_CONSTANT = 0.0000000000000000e+00f;
#define TT_SPECIAL_NAN 0
#define TT_SPECIAL_POS_INF 5
#define TT_SPECIAL_NEG_INF 3
#define TT_SPECIAL_POS_ZERO 5
#define TT_SPECIAL_NEG_ZERO 5

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_core_total.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_prepare.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_action_coordinate.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_raw_class_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_target_special_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_encoded_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_zone_gradient_finalize.inc"
#define DST_COEFF_ELIGIBLE 1
constexpr bool kDstCoeffApply = true;
namespace dstcoeff {
struct Layout {
    static constexpr uint32_t kSegments = NUM_SEGMENTS;
    static constexpr uint32_t kCoefficientOffset = NUM_SEGMENTS + 1;
    static constexpr uint32_t kCoefficientsPerSegment = POLY_DEGREE + 1;
    static constexpr uint32_t kRowBase = 64;
    static constexpr uint32_t degree(uint32_t s) { return SEGMENT_DEGREES[s]; }
    static constexpr int rows[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,
                                   8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, -1, -1, -1, -1,
                                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr int park_row(uint32_t i) { return rows[i]; }
    static constexpr float lut(uint32_t i) { return LUT_DATA[i]; }
};
static_assert(Layout::kRowBase + 20u <= 128u);
using Transport = ::sfpi::dense_polynomial_transport<Layout>;
__attribute__((always_inline)) inline void park_all() {
    Transport::park_boundaries<1>();
    Transport::park_coeffs<0>();
}
}  // namespace dstcoeff
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_dst_tile.inc"
inline void tile() { piecewise_generic_lut_dst_coeff<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(LUT_DATA); }
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_gradient_finalize.inc"

}  // namespace ttpoly_generated::CeluBwBf16Config_detail
namespace ttpoly_generated::CeluBwBf16Config_execution {
namespace sfpi = ::ttpoly_generated::CeluBwBf16Config_detail;
using namespace sfpi;
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_gradient_finalization_owner.inc"
inline void init() {
#if defined(ARCH_WORMHOLE) && defined(TRISC_MATH)
    ckernel::llk_math_sfpu_init_once();
#endif
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_reciprocal_init.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_exp_pool_init.inc"
}
}  // namespace ttpoly_generated::CeluBwBf16Config_execution
namespace ttpoly_generated {
struct CeluBwBf16Config {
    static constexpr bool needs_gradient = !CeluBwBf16Config_execution::gradient_finalized_in_evaluator;
    static inline void init() { CeluBwBf16Config_execution::init(); }
    static inline void tile() { CeluBwBf16Config_detail::tile(); }
    struct Gradient {
        static inline void tile() { CeluBwBf16Config_detail::fuse_grad_mul(); }
    };
};
}  // namespace ttpoly_generated
#pragma pop_macro("USE_DUAL_EVAL")
#pragma pop_macro("USE_BF16")
#pragma pop_macro("TT_WH_DST_COEFF_ROW_UNROLL")
#pragma pop_macro("TT_WH_DENSE_DST_COEFF_STORE")
#pragma pop_macro("TT_WH_CORE_SAME_ROW_GRAD_FINALIZE")
#pragma pop_macro("TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED")
#pragma pop_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#pragma pop_macro("TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT")
#pragma pop_macro("TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR")
#pragma pop_macro("TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT")
#pragma pop_macro("TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR")
#pragma pop_macro("TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL")
#pragma pop_macro("TT_SPECIAL_VALUE_POLICY")
#pragma pop_macro("TT_SPECIAL_POS_ZERO")
#pragma pop_macro("TT_SPECIAL_POS_INF")
#pragma pop_macro("TT_SPECIAL_NEG_ZERO")
#pragma pop_macro("TT_SPECIAL_NEG_INF")
#pragma pop_macro("TT_SPECIAL_NAN")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_FORM")
#pragma pop_macro("TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1")
#pragma pop_macro("TT_ACT_EVAL_KIND")
#pragma pop_macro("SEG7_DEGREE")
#pragma pop_macro("SEG6_DEGREE")
#pragma pop_macro("SEG5_DEGREE")
#pragma pop_macro("SEG4_DEGREE")
#pragma pop_macro("SEG3_DEGREE")
#pragma pop_macro("SEG2_DEGREE")
#pragma pop_macro("SEG1_DEGREE")
#pragma pop_macro("SEG0_DEGREE")
#pragma pop_macro("HAS_SEGMENT_DEGREES")
#pragma pop_macro("FUSE_GRAD_MUL")
#pragma pop_macro("EVAL_METHOD_POLY_CASCADE")
#pragma pop_macro("EMBEDDED_LUT")
#pragma pop_macro("DST_COEFF_STORE")
#pragma pop_macro("DST_COEFF_ELIGIBLE")
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_config_tile.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_celu_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::CeluBwBf16Config, ITERATIONS>();
}
inline void init_celu_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_config_tile<ttpoly_generated::CeluBwBf16Config>();
}
template <int ITERATIONS = 32>
inline void calculate_celu_bw_gradient_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::CeluBwBf16Config::Gradient, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
