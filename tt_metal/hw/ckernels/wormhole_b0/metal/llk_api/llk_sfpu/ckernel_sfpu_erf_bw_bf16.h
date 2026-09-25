// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <array>
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_horner.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_tti_replay.h"
namespace sfpi {
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_min_max.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_mirrored_terminals.h"
}  // namespace sfpi
#if !defined(ARCH_WORMHOLE)
#error "square-affine target differs from its selected source"
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_X) ||   \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_NEGATE_OUTPUT) || defined(ASYMPTOTIC_REGION_LEFT) || \
    defined(ASYMPTOTIC_REGION_RIGHT) || defined(BASIS_AFFINE_EVEN) || defined(BASIS_CLAMP_MAX) ||                      \
    defined(BASIS_INPUT_ABS_X) || defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_MUL_ABS_X_BEFORE_POST) ||             \
    defined(BASIS_MUL_SQRT_1_MINUS_ABS) || defined(BASIS_POST_REFLECT_PI) || defined(BASIS_POST_SIGN_X) ||             \
    defined(BASIS_RIGHT_TAIL_ABS_AFFINE) || defined(BASIS_RIGHT_TAIL_IDENTITY) || defined(DISABLE_ADAPTIVE_DEGREE) ||  \
    defined(DOMAIN_ACTION_DISABLE) || defined(DST_COEFF_DISABLE) || defined(DST_COEFF_ELIGIBLE) ||                     \
    defined(DST_COEFF_PROBE) || defined(DST_COEFF_STORE) || defined(DUAL_EVAL_DISABLE) ||                              \
    defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1) || defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) ||      \
    defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) || defined(EVAL_METHOD_TAN_STANDALONE) ||            \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) || defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) ||                   \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || defined(EXP_HW_COMPOSE_SIGMOID) ||                                \
    defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) ||                    \
    defined(HAS_CRITICAL_POINT) || defined(HAS_SEGMENT_DEGREES) || defined(POLY_PARITY_EVEN) ||                        \
    defined(POLY_PARITY_ODD) || defined(POLY_TTI_DISABLE) || defined(POW_HW_RECIPROCAL) ||                             \
    defined(RANGE_REDUCTION_CBRT) || defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_LOG) ||                   \
    defined(RANGE_REDUCTION_TAN) || defined(RANGE_REDUCTION_TRIG) || defined(RATIONAL_RECIPROCAL_ONE_ITER) ||          \
    defined(RATIONAL_RECIPROCAL_ZERO_ITER) || defined(RATIONAL_TTI_DISABLE) ||                                         \
    defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD) ||                       \
    defined(REDUCE_EXP_COMPOSE_SIGMOID) || defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) ||                              \
    defined(SPECIAL_VALUE_DISABLE) || defined(TT_ABS_RESIDUAL_AFFINE_1_NEEDS_RECIPROCAL) ||                            \
    defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP) || defined(TT_DOMAIN_ACTION_RAW_TERMINAL_ENVELOPE) ||           \
    defined(TT_DOMAIN_ACTION_TERMINAL_INGRESS_ONLY) || defined(TT_DOMAIN_ACTION_WH_ORDERED_INGRESS) ||                 \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) ||                                                     \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_INLINE_PROGRAM_GRADIENT_SCALE) ||            \
    defined(TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT) || defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) ||                  \
    defined(TT_INLINE_PROGRAM_NEEDS_RECIPROCAL) || defined(TT_LINEAR_BF16_UNDERFLOW_COUNT) ||                          \
    defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND) || defined(TT_SELECTED_CORE_AGGREGATE_STIRLING) ||          \
    defined(TT_SELECTED_CORE_BRIDGE_DIRECT_LOG_TAIL_1) || defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1) ||   \
    defined(TT_SELECTED_CORE_EXP2_SHIFTED) || defined(TT_SELECTED_CORE_EXP_TAIL_ABOVE) ||                              \
    defined(TT_SELECTED_CORE_GAUSSIAN_MILLS_TAIL_1) || defined(TT_SELECTED_CORE_NEEDS_RECIPROCAL) ||                   \
    defined(TT_SELECTED_CORE_ONE_SIDED_CODY_EXP_TAIL_1) || defined(TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1) ||           \
    defined(TT_SELECTED_CORE_SCALED_EXP_CORRECTION_TAIL_1) || defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1) ||   \
    defined(TT_SELECTED_CORE_SIGMOID_PRODUCT_DERIVATIVE_TAIL_1) ||                                                     \
    defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1) ||                                                       \
    defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1) || defined(TT_SELECTED_CORE_SYMMETRIC_EXP_TAIL_1) ||         \
    defined(TT_SELECTED_CORE_TOTAL_FORM) || defined(TT_SELECTED_CORE_TOTAL_INPLACE_FACTORED_OVERLAY_V1) ||             \
    defined(TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL) || defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) ||      \
    defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW) || defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) ||        \
    defined(TT_SELECTED_WH_AGGREGATE_STIRLING) || defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL) ||                     \
    defined(TT_SELECTED_ZONE_GRADIENT) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) ||                             \
    defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO) || defined(TT_SPECIAL_COMPARE_NEG_INF) ||                           \
    defined(TT_SPECIAL_COMPARE_POS_INF) || defined(TT_SPECIAL_COMPARE_POS_ZERO) ||                                     \
    defined(TT_SQUARE_DECAY_NEEDS_RECIPROCAL) || defined(TT_TARGET_BH_BF16_ACTION_COORDINATE) ||                       \
    defined(TT_TARGET_BH_BF16_PACK_RELU_RAW_NEG_NAN_REPAIR) ||                                                         \
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
    defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS) || defined(TT_WH_FACTORED_CORE_MIRROR_FOLD) ||                          \
    defined(TT_WH_NORMALIZED_LOG1P_TERMINALS) || defined(TT_WH_RAW_NAN_UNION) || defined(TT_WH_SYMMETRIC_EXP_POOL)
#error "square-affine cannot inherit alternate numerical selectors"
#endif
#pragma push_macro("ASYMPTOTIC_FACTOR_EXP_QUADRATIC")
#undef ASYMPTOTIC_FACTOR_EXP_QUADRATIC
#pragma push_macro("ASYMPTOTIC_REGION_ALL")
#undef ASYMPTOTIC_REGION_ALL
#pragma push_macro("EMBEDDED_LUT")
#undef EMBEDDED_LUT
#pragma push_macro("EVAL_METHOD_POLY_CASCADE")
#undef EVAL_METHOD_POLY_CASCADE
#pragma push_macro("FUSE_GRAD_MUL")
#undef FUSE_GRAD_MUL
#pragma push_macro("POLY_TTI_SHAPE_PLAIN")
#undef POLY_TTI_SHAPE_PLAIN
#pragma push_macro("TT_ACT_EVAL_KIND")
#undef TT_ACT_EVAL_KIND
#pragma push_macro("TT_ASYMPTOTIC_END_IF")
#undef TT_ASYMPTOTIC_END_IF
#pragma push_macro("TT_ASYMPTOTIC_IF")
#undef TT_ASYMPTOTIC_IF
#pragma push_macro("TT_DOMAIN_ACTION_PROGRAM")
#undef TT_DOMAIN_ACTION_PROGRAM
#pragma push_macro("TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS")
#undef TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS
#pragma push_macro("TT_DOMAIN_ACTION_TERMINAL_ACTION_ONLY")
#undef TT_DOMAIN_ACTION_TERMINAL_ACTION_ONLY
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
#define USE_BF16 1
#define FUSE_GRAD_MUL 1
#define POLY_TTI_SHAPE_PLAIN 1
#define USE_DUAL_EVAL 1
namespace ttpoly_generated::ErfBwBf16Config_detail {
using namespace ::sfpi;
using ::sfpi::DataLayout;
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated by run_csv.sh
// Degree 1, 1 segments, range [-10.0, 10.0]

#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = 1;
constexpr uint32_t NUM_SEGMENTS = 1;

constexpr float INPUT_MIN = -1.0000000000000000e+01f;
constexpr float INPUT_MAX = 1.0000000000000000e+01f;

constexpr uint32_t LUT_SIZE_BF16 = 4;
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {
    {-1.0000000000000000e+01f, 1.0000000000000000e+01f, 1.1283752628727044e+00f, 6.1106666499469343e-16f}};

constexpr uint32_t LUT_SIZE_FP32 = 4;
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {
    {-1.0000000000000000e+01f, 1.0000000000000000e+01f, 1.1283752628727044e+00f, 6.1106666499469343e-16f}};

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

// Asymptotic factoring: exp(-x^2)
#define ASYMPTOTIC_FACTOR_EXP_QUADRATIC
constexpr float ASYMPTOTIC_EXP_ARG_SCALE = -1.0000000000000000e+00f;
constexpr float ASYMPTOTIC_SCALE = 1.0000000000000000e+00f;
#define ASYMPTOTIC_REGION_ALL

// Typed domain actions.
#define TT_DOMAIN_ACTION_PROGRAM
struct TtDomainActionRecord {
    float bound;
    uint8_t direction;
    uint8_t inclusive;
    uint8_t action_kind;   // 0=constant, 1=identity, 2=affine, 3=class, 4=signed-inf
    uint8_t return_class;  // 0=NaN, 1=+Inf, 2=-Inf, 3=+0, 4=-0
    float value;
    float scale;
    float bias;
};
constexpr uint32_t TT_DOMAIN_ACTION_COUNT = 2;
constexpr std::array<TtDomainActionRecord, TT_DOMAIN_ACTION_COUNT> TT_DOMAIN_ACTION_DATA = {
    {{-9.3750000000000000e+00f, 0, 1, 0, 0, 0.0000000000000000e+00f, 0.0000000000000000e+00f, 0.0000000000000000e+00f},
     {9.3750000000000000e+00f, 1, 1, 0, 0, 0.0000000000000000e+00f, 0.0000000000000000e+00f, 0.0000000000000000e+00f}}};
#define TT_DOMAIN_ACTION_TERMINAL_ACTION_ONLY 1
#define TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS
constexpr float TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND = 9.3750000000000000e+00f;
constexpr float TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE = 0.0000000000000000e+00f;
constexpr uint8_t TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE = 1;

// Declared special-value policy from the typed activation specification.
// 0=NaN 1=+Inf 2=-Inf 3=+0 4=-0 5=finite_other(pass through)
#define TT_SPECIAL_VALUE_POLICY
#define TT_SPECIAL_NAN 0
#define TT_SPECIAL_POS_INF 3
#define TT_SPECIAL_NEG_INF 3
#define TT_SPECIAL_POS_ZERO 5
#define TT_SPECIAL_NEG_ZERO 5

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_core_total.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_prepare.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_action_coordinate.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_raw_class_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_target_special_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_symmetric_constant.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_encoded_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_asymptotic_exp.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_asymptotic_factor.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_dual_tile.inc"
inline void tile() { piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(LUT_DATA); }

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_gradient_finalize.inc"
}  // namespace ttpoly_generated::ErfBwBf16Config_detail
namespace ttpoly_generated::ErfBwBf16Config_initialization {
namespace sfpi = ::ttpoly_generated::ErfBwBf16Config_detail;
inline void init() {
#if defined(ARCH_WORMHOLE) && defined(TRISC_MATH)
    ckernel::llk_math_sfpu_init_once();
#endif
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_reciprocal_init.inc"
}
}  // namespace ttpoly_generated::ErfBwBf16Config_initialization
namespace ttpoly_generated::ErfBwBf16Config_execution {
namespace sfpi = ::ttpoly_generated::ErfBwBf16Config_detail;
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_gradient_finalization_owner.inc"
}  // namespace ttpoly_generated::ErfBwBf16Config_execution
namespace ttpoly_generated {
struct ErfBwBf16Config {
    static constexpr bool needs_gradient = !ErfBwBf16Config_execution::gradient_finalized_in_evaluator;
    static inline void init() { ErfBwBf16Config_initialization::init(); }
    static inline void tile() { ErfBwBf16Config_detail::tile(); }
    struct Gradient {
        static inline void tile() { ErfBwBf16Config_detail::fuse_grad_mul(); }
    };
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
#pragma pop_macro("TT_DOMAIN_ACTION_TERMINAL_ACTION_ONLY")
#pragma pop_macro("TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS")
#pragma pop_macro("TT_DOMAIN_ACTION_PROGRAM")
#pragma pop_macro("TT_ASYMPTOTIC_IF")
#pragma pop_macro("TT_ASYMPTOTIC_END_IF")
#pragma pop_macro("TT_ACT_EVAL_KIND")
#pragma pop_macro("POLY_TTI_SHAPE_PLAIN")
#pragma pop_macro("FUSE_GRAD_MUL")
#pragma pop_macro("EVAL_METHOD_POLY_CASCADE")
#pragma pop_macro("EMBEDDED_LUT")
#pragma pop_macro("ASYMPTOTIC_REGION_ALL")
#pragma pop_macro("ASYMPTOTIC_FACTOR_EXP_QUADRATIC")
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_config_tile.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_erf_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::ErfBwBf16Config, ITERATIONS>();
}
inline void init_erf_bw_tt_poly_bf16() { ckernel::sfpu::ttpoly::init_config_tile<ttpoly_generated::ErfBwBf16Config>(); }
template <int ITERATIONS = 32>
inline void calculate_erf_bw_gradient_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::ErfBwBf16Config::Gradient, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
