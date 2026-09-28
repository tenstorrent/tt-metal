// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#if defined(ARCH_BLACKHOLE)
#include "ckernel_sfpu_conversions.h"
#endif
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_horner.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_tti_replay.h"
namespace sfpi {
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_min_max.h"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_mirrored_terminals.h"
}  // namespace sfpi
#if !defined(ARCH_WORMHOLE)
#error "square-affine target differs from its selected source"
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) ||                               \
    defined(ASYMPTOTIC_FACTOR_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_X) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || \
    defined(BASIS_AFFINE_EVEN) || defined(BASIS_CLAMP_MAX) || defined(BASIS_INPUT_ABS_X) ||                            \
    defined(BASIS_LEFT_TAIL_ZERO) || defined(BASIS_MUL_ABS_X_BEFORE_POST) || defined(BASIS_MUL_SQRT_1_MINUS_ABS) ||    \
    defined(BASIS_POST_REFLECT_PI) || defined(BASIS_POST_SIGN_X) || defined(BASIS_RIGHT_TAIL_ABS_AFFINE) ||            \
    defined(BASIS_RIGHT_TAIL_IDENTITY) || defined(DISABLE_ADAPTIVE_DEGREE) || defined(DOMAIN_ACTION_DISABLE) ||        \
    defined(DST_COEFF_DISABLE) || defined(DST_COEFF_ELIGIBLE) || defined(DST_COEFF_PROBE) ||                           \
    defined(DST_COEFF_STORE) || defined(DUAL_EVAL_DISABLE) || defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1) || \
    defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) ||                                                               \
    defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) || defined(EVAL_METHOD_TAN_STANDALONE) ||            \
    defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) || defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) ||                   \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || defined(EXP_HW_COMPOSE_SIGMOID) ||                                \
    defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) ||                    \
    defined(HAS_CRITICAL_POINT) || defined(HAS_SEGMENT_DEGREES) || defined(POLY_PARITY_EVEN) ||                        \
    defined(POLY_PARITY_ODD) || defined(POLY_TTI_SHAPE_LOG_SQUARE) || defined(POLY_TTI_SHAPE_PARITY_EVEN) ||           \
    defined(POW_HW_RECIPROCAL) || defined(RANGE_REDUCTION_CBRT) || defined(RANGE_REDUCTION_EXP) ||                     \
    defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_RECIP_COMPLEMENT) || defined(RANGE_REDUCTION_TAN) ||       \
    defined(RANGE_REDUCTION_TRIG) || defined(RATIONAL_RECIPROCAL_ONE_ITER) ||                                          \
    defined(RATIONAL_RECIPROCAL_ZERO_ITER) || defined(RATIONAL_TTI_DISABLE) || defined(REDUCE_EXP_BASE2) ||            \
    defined(REDUCE_EXP_COMPOSE_ELU) || defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) ||                                  \
    defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD) || defined(REDUCE_EXP_COMPOSE_SELU) ||                                  \
    defined(REDUCE_EXP_COMPOSE_SIGMOID) || defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) ||                              \
    defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED) || defined(REDUCE_LOG_COMPOSE_SQRT2_CENTERED_SPLIT) ||                  \
    defined(REDUCE_RECIP_COMPLEMENT) || defined(SCALAR_ROW_UNROLL_DISABLE) || defined(SPECIAL_VALUE_DISABLE) ||        \
    defined(TT_ABS_RESIDUAL_AFFINE_1_NEEDS_RECIPROCAL) || defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP) ||        \
    defined(TT_DOMAIN_ACTION_PROGRAM) || defined(TT_DOMAIN_ACTION_RAW_TERMINAL_ENVELOPE) ||                            \
    defined(TT_DOMAIN_ACTION_TERMINAL_INGRESS_ONLY) || defined(TT_DOMAIN_ACTION_WH_ORDERED_INGRESS) ||                 \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) ||                                                     \
    defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_INLINE_PROGRAM_GRADIENT_SCALE) ||            \
    defined(TT_INLINE_PROGRAM_GRAD_ZERO_MASK_SELECT) || defined(TT_INLINE_PROGRAM_INPLACE_OVERLAY) ||                  \
    defined(TT_INLINE_PROGRAM_NEEDS_RECIPROCAL) || defined(TT_INLINE_PROGRAM_SELECTED_FACTOR_INTRINSIC_CLAMP) ||       \
    defined(TT_LINEAR_BF16_UNDERFLOW_COUNT) || defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND) ||               \
    defined(TT_SELECTED_CORE_AGGREGATE_STIRLING) || defined(TT_SELECTED_CORE_BRIDGE_DIRECT_LOG_TAIL_1) ||              \
    defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1) || defined(TT_SELECTED_CORE_EXP_TAIL_ABOVE) ||             \
    defined(TT_SELECTED_CORE_GAUSSIAN_MILLS_TAIL_1) || defined(TT_SELECTED_CORE_NEEDS_RECIPROCAL) ||                   \
    defined(TT_SELECTED_CORE_ONE_SIDED_CODY_EXP_TAIL_1) || defined(TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1) ||           \
    defined(TT_SELECTED_CORE_SCALED_EXP_CORRECTION_TAIL_1) || defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1) ||   \
    defined(TT_SELECTED_CORE_SIGMOID_PRODUCT_DERIVATIVE_TAIL_1) ||                                                     \
    defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1) || defined(TT_SELECTED_CORE_SYMMETRIC_EXP_TAIL_1) ||         \
    defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) ||                                                          \
    defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_INTRINSIC_RAW_CLOSURE) ||                                               \
    defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) || defined(TT_SELECTED_WH_AGGREGATE_STIRLING) ||        \
    defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL) || defined(TT_SELECTED_ZONE_GRADIENT) ||                             \
    defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO) ||                 \
    defined(TT_SPECIAL_COMPARE_NEG_INF) || defined(TT_SPECIAL_COMPARE_POS_INF) ||                                      \
    defined(TT_SPECIAL_COMPARE_POS_ZERO) || defined(TT_SQUARE_DECAY_NEEDS_RECIPROCAL) ||                               \
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
    defined(TT_TARGET_BH_FP32_RAW_NEG_ZERO_DISCRIMINATOR) || defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS) ||             \
    defined(TT_WH_NORMALIZED_LOG1P_TERMINALS) || defined(TT_WH_RAW_NAN_UNION) ||                                       \
    defined(TT_WH_REDUCED_POLY_UNROLL32) || defined(TT_WH_SIGNED_ABS_COEFFICIENT_STORE) ||                             \
    defined(TT_WH_SIGNED_ABS_UNROLL32) || defined(TT_WH_SYMMETRIC_EXP_POOL)
#error "square-affine cannot inherit alternate numerical selectors"
#endif
#pragma push_macro("EMBEDDED_LUT")
#undef EMBEDDED_LUT
#pragma push_macro("EVAL_METHOD_POLY_CASCADE")
#undef EVAL_METHOD_POLY_CASCADE
#pragma push_macro("FUSE_GRAD_MUL")
#undef FUSE_GRAD_MUL
#pragma push_macro("POLY_TTI_DISABLE")
#undef POLY_TTI_DISABLE
#pragma push_macro("POLY_TTI_SHAPE_PLAIN")
#undef POLY_TTI_SHAPE_PLAIN
#pragma push_macro("TT_ACT_EVAL_KIND")
#undef TT_ACT_EVAL_KIND
#pragma push_macro("TT_SELECTED_COMPONENT_SFPI_LOOP_ROW")
#undef TT_SELECTED_COMPONENT_SFPI_LOOP_ROW
#pragma push_macro("TT_SELECTED_CORE_EXP2_SHIFTED")
#undef TT_SELECTED_CORE_EXP2_SHIFTED
#pragma push_macro("TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1")
#undef TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1
#pragma push_macro("TT_SELECTED_CORE_TOTAL_FORM")
#undef TT_SELECTED_CORE_TOTAL_FORM
#pragma push_macro("TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL")
#undef TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL
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
#pragma push_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#undef TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT
#pragma push_macro("TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED")
#undef TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED
#pragma push_macro("TT_WH_CORE_SAME_ROW_GRAD_FINALIZE")
#undef TT_WH_CORE_SAME_ROW_GRAD_FINALIZE
#pragma push_macro("USE_BF16")
#undef USE_BF16
#define USE_BF16 1
#define FUSE_GRAD_MUL 1
#define POLY_TTI_SHAPE_PLAIN 1

namespace ttpoly_generated::GeluBwBf16Config_detail {
using namespace ::sfpi;
using ::sfpi::DataLayout;
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated by run_csv.sh
// Degree 8, 1 segments, range [0.0, 12.25]

#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = 8;
constexpr uint32_t NUM_SEGMENTS = 1;

constexpr float INPUT_MIN = 0.0000000000000000e+00f;
constexpr float INPUT_MAX = 1.2250000000000000e+01f;

constexpr uint32_t LUT_SIZE_BF16 = 11;
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {
    {0.0000000000000000e+00f,
     1.2250000000000000e+01f,
     7.9788136482238770e-01f,
     -2.6588755846023560e-01f,
     5.9638097882270813e-02f,
     -9.3036592006683350e-03f,
     1.0614653583616018e-03f,
     -8.7852131400723010e-05f,
     4.9631953515927307e-06f,
     -1.6895057797228219e-07f,
     2.5827593397309556e-09f}};

constexpr uint32_t LUT_SIZE_FP32 = 11;
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {
    {0.0000000000000000e+00f,
     1.2250000000000000e+01f,
     7.9788136482238770e-01f,
     -2.6588755846023560e-01f,
     5.9638097882270813e-02f,
     -9.3036592006683350e-03f,
     1.0614653583616018e-03f,
     -8.7852131400723010e-05f,
     4.9631953515927307e-06f,
     -1.6895057797228219e-07f,
     2.5827593397309556e-09f}};

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
#define TT_WH_CORE_SAME_ROW_GRAD_FINALIZE 1

// Declared special-value policy from the typed activation specification.
// 0=NaN 1=+Inf 2=-Inf 3=+0 4=-0 5=finite_other(pass through)
#define TT_SPECIAL_VALUE_POLICY
#define TT_SPECIAL_NAN 3
#define TT_SPECIAL_POS_INF 5
#define TT_SPECIAL_NEG_INF 3
#define TT_SPECIAL_POS_ZERO 5
#define TT_SPECIAL_NEG_ZERO 5
#define TT_SELECTED_CORE_TOTAL_FORM 1
#define TT_SELECTED_CORE_TOTAL_SINGLE_ROW 1
#define TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1 1
#define TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL 1
#define POLY_TTI_DISABLE 1
#define TT_SELECTED_COMPONENT_SFPI_LOOP_ROW 1
constexpr float TT_SELECTED_CORE_LOWER = -3.0000000000000000e+00f;
constexpr float TT_SELECTED_CORE_UPPER = 3.5000000000000000e+00f;
constexpr float TT_SELECTED_CORE_NEGATIVE_TERMINAL = -1.3341882705688477e+01f;
constexpr float TT_SELECTED_CORE_POSITIVE_IDENTITY = 3.5000000000000000e+00f;
constexpr float TT_SELECTED_CORE_EXPONENT_SCALE = -5.0000000000000000e-01f;
constexpr float TT_SELECTED_CORE_CORRECTION_MIN = 5.0000000000000001e-03f;
constexpr float TT_SELECTED_CORE_CORRECTION_MAX = 1.2500000000000000e-01f;
constexpr float TT_SELECTED_CORE_CORRECTION_SCALE = 2.4933892525089544e-02f;
constexpr uint32_t TT_SELECTED_CORE_RECIPROCAL_ITERS = 1u;
constexpr uint32_t TT_SELECTED_CORE_CORRECTION_DEGREE = 2u;
constexpr float TT_SELECTED_CORE_CORRECTION_COEFFS[] = {
    9.9986010789871216e-01f, -9.8491287231445312e-01f, 6.4935469627380371e-01f};
constexpr uint32_t TT_SELECTED_CORE_EXP_DEGREE = 3u;
constexpr float TT_SELECTED_CORE_EXP_COEFFS[] = {
    1.0000000000000000e+00f, 6.9556409120559692e-01f, 2.2616928815841675e-01f, 7.8141160309314728e-02f};
constexpr float TT_SELECTED_CORE_EXP2_MULT = 1.4426950408889634e+00f;
constexpr int TT_SELECTED_CORE_EXP2_OUTPUT_SHIFT = 4;
constexpr int TT_SELECTED_CORE_EXP2_BIAS = 131;
#define TT_SELECTED_CORE_EXP2_SHIFTED 1
constexpr uint32_t TT_SELECTED_CORE_OUTPUT_SCALE_BITS = 0x3f800000u;
constexpr uint32_t TT_SELECTED_CORE_OUTPUT_BIAS_BITS = 0x3f000000u;

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_core_total.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_prepare.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_action_coordinate.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_raw_class_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_target_special_policy.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_domain_finalize.inc"
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_encoded_domain_finalize.inc"
#if defined(ARCH_BLACKHOLE)
constexpr uint32_t kPolyTtiCoeffOff = NUM_SEGMENTS + 1;
constexpr float poly_tti_c(uint32_t k) { return LUT_DATA[kPolyTtiCoeffOff + k]; }
constexpr uint32_t poly_tti_addend_bits(uint32_t j) {
    return __builtin_bit_cast(uint32_t, poly_tti_c(POLY_DEGREE - 2u - j));
}
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_square_affine_contract.inc"
static_assert(kPolyTtiSquareAffine1x);
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_square_affine_replay.inc"
inline void tile() { poly_tti_replay_tile_square_affine(); }
constexpr uint32_t kPolyTtiPrgmWriteMask = 6u, kPolyTtiPrgm0Mask = 1u;
constexpr bool kPolyTtiPlain2x = false, kPolyTtiParityEven2x = false;
#else
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_single_tile.inc"
inline void tile() { piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(LUT_DATA); }
#endif

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_gradient_finalize.inc"
}  // namespace ttpoly_generated::GeluBwBf16Config_detail
namespace ttpoly_generated::GeluBwBf16Config_initialization {
namespace sfpi = ::ttpoly_generated::GeluBwBf16Config_detail;
inline void init() {
#if defined(ARCH_WORMHOLE) && defined(TRISC_MATH)
    ckernel::llk_math_sfpu_init_once();
#endif
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_selected_reciprocal_init.inc"
#if defined(ARCH_BLACKHOLE)
    using sfpi::POLY_DEGREE;
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_polynomial_replay_init.inc"
#endif
}
}  // namespace ttpoly_generated::GeluBwBf16Config_initialization
namespace ttpoly_generated::GeluBwBf16Config_execution {
namespace sfpi = ::ttpoly_generated::GeluBwBf16Config_detail;
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_gradient_finalization_owner.inc"
}  // namespace ttpoly_generated::GeluBwBf16Config_execution
namespace ttpoly_generated {
struct GeluBwBf16Config {
    static constexpr bool needs_gradient = !GeluBwBf16Config_execution::gradient_finalized_in_evaluator;
    static inline void init() { GeluBwBf16Config_initialization::init(); }
    static inline void tile() { GeluBwBf16Config_detail::tile(); }
    struct Gradient {
        static inline void tile() { GeluBwBf16Config_detail::fuse_grad_mul(); }
    };
};
}  // namespace ttpoly_generated
#pragma pop_macro("USE_BF16")
#pragma pop_macro("TT_WH_CORE_SAME_ROW_GRAD_FINALIZE")
#pragma pop_macro("TT_TARGET_WH_RAW_NEG_EXPONENT_ZERO_FUSED")
#pragma pop_macro("TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT")
#pragma pop_macro("TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL")
#pragma pop_macro("TT_SPECIAL_VALUE_POLICY")
#pragma pop_macro("TT_SPECIAL_POS_ZERO")
#pragma pop_macro("TT_SPECIAL_POS_INF")
#pragma pop_macro("TT_SPECIAL_NEG_ZERO")
#pragma pop_macro("TT_SPECIAL_NEG_INF")
#pragma pop_macro("TT_SPECIAL_NAN")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_SINGLE_ROW")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL")
#pragma pop_macro("TT_SELECTED_CORE_TOTAL_FORM")
#pragma pop_macro("TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1")
#pragma pop_macro("TT_SELECTED_CORE_EXP2_SHIFTED")
#pragma pop_macro("TT_SELECTED_COMPONENT_SFPI_LOOP_ROW")
#pragma pop_macro("TT_ACT_EVAL_KIND")
#pragma pop_macro("POLY_TTI_SHAPE_PLAIN")
#pragma pop_macro("POLY_TTI_DISABLE")
#pragma pop_macro("FUSE_GRAD_MUL")
#pragma pop_macro("EVAL_METHOD_POLY_CASCADE")
#pragma pop_macro("EMBEDDED_LUT")
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_config_tile.h"
#define TT_POLY_GELU_BW_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_gelu_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::GeluBwBf16Config, ITERATIONS>();
}
inline void init_gelu_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_config_tile<ttpoly_generated::GeluBwBf16Config>();
}
template <int ITERATIONS = 32>
inline void calculate_gelu_bw_gradient_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::GeluBwBf16Config::Gradient, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
