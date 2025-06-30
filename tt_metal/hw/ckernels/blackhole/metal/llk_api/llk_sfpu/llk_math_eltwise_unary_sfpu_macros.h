// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// For ops that require only the init function
#define SFPU_UNARY_KERNEL_INIT_ONLY(OP)                                \
    template <bool APPROXIMATE>                                        \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {            \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(); \
    }

// For ops that require only an init function with a custom SfpuType and init callback
#define SFPU_INIT_ONLY_WITH_TYPE(NAME, TYPE, INIT_CB)                                        \
    template <bool APPROXIMATE>                                                              \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>(INIT_CB<APPROXIMATE>); \
    }

// For ops that require only the compute function
#define SFPU_UNARY_KERNEL_NO_INIT(OP)                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode);                          \
    }

// For ops with extra arguments that require a custom compute callback instead of the default calculate_<OP>
#define SFPU_UNARY_PARAMS_KERNEL_WITH_CUSTOM_CALC(OP, MODE, CALC_CB, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                   \
    template <bool APPROXIMATE>                                                                                        \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                                      \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {                                     \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(CALC_CB<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For init-only kernels using SfpuType::unused
#define SFPU_UNARY_KERNEL_INIT_UNUSED(OP)                                  \
    template <bool APPROXIMATE>                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(); \
    }

// For rounding ops with explicit ITERATIONS/USE_FP32 template params
#define SFPU_ROUNDING_OP_KERNEL(NAME)                                                                        \
    template <bool APPROXIMATE, int ITERATIONS = 8, bool USE_FP32 = false>                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index, int vector_mode = (int)VectorMode::RC) {  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                     \
            ckernel::sfpu::_calculate_##NAME##_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode); \
    }

// For float32 rounding variants (USE_FP32 = true)
#define SFPU_ROUNDING_OP_KERNEL_FLOAT32(NAME)                                                                         \
    template <bool APPROXIMATE, int ITERATIONS = 8, bool USE_FP32 = true>                                             \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_float32(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                              \
            ckernel::sfpu::_calculate_##NAME##_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode);          \
    }

// For truncation ops with no extra template params
#define SFPU_TRUNC_OP_KERNEL(NAME)                                                                          \
    template <bool APPROXIMATE>                                                                             \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                    \
            ckernel::sfpu::_calculate_##NAME##_<APPROXIMATE>, dst_index, vector_mode);                      \
    }

// For float32 truncation variant (USE_FP32 = true)
#define SFPU_TRUNC_OP_KERNEL_FLOAT32(NAME)                                                                            \
    template <bool APPROXIMATE>                                                                                       \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_float32(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                              \
            ckernel::sfpu::_calculate_##NAME##_<APPROXIMATE, true>, dst_index, vector_mode);                          \
    }

// For round op with int decimals arg and custom callback
#define SFPU_ROUND_WITH_DECIMALS_KERNEL(OP)                                                    \
    template <bool APPROXIMATE>                                                                \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                              \
        uint dst_index, int decimals, int vector_mode = (int)VectorMode::RC) {                 \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                       \
            ckernel::sfpu::_calculate_##OP##_<APPROXIMATE>, dst_index, vector_mode, decimals); \
    }

// For unary ops that also need an _int32 variant
#define SFPU_UNARY_INT32_KERNEL(OP)                                                                               \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##OP##_int32(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                          \
            ckernel::sfpu::calculate_##OP##_int32<APPROXIMATE>, dst_index, vector_mode);                          \
    }

// For ops with extra compute template parameter (e.g., bool flag) in addition to APPROXIMATE
#define SFPU_UNARY_PARAMS_KERNEL_WITH_EXTRA_TEMPLATE(                                      \
    OP, MODE, CALC_CB, EXTRA_TEMPLATE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                     \
    template <bool APPROXIMATE, EXTRA_TEMPLATE>                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                          \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {         \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                   \
            CALC_CB<APPROXIMATE, EXTRA_TEMPLATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For ops where compute takes extra args, and no special init
#define SFPU_UNARY_PARAMS_KERNEL_ONLY_COMPUTE(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)          \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {               \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For a pair of compute‐only ops that switch on an enum
#define SFPU_DIM_DUALTYPE_SWITCH_KERNEL(OP, ENUM, TYPE0A, TYPE0B, CALC0, MODE0, CALC1, MODE1)                       \
    template <bool APPROXIMATE>                                                                                     \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, ENUM dim, int vector_mode = (int)VectorMode::RC) { \
        if (dim == ENUM::TYPE0A || dim == ENUM::TYPE0B) {                                                           \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                        \
                ckernel::sfpu::CALC0<APPROXIMATE>, dst_index, vector_mode);                                         \
        } else if (dim == ENUM::MODE1) {                                                                            \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                        \
                ckernel::sfpu::CALC1<APPROXIMATE>, dst_index, vector_mode);                                         \
        }                                                                                                           \
    }

// For ops with two implementations that select compute callback based on an enum argument
#define SFPU_DIM_DUALTYPE_SWITCH_KERNEL(OP, ENUM, TYPE0A, TYPE0B, CALC0, MODE0, CALC1, MODE1) \
    template <bool APPROXIMATE>                                                               \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                             \
        uint dst_index, ENUM data_format, int vector_mode = (int)VectorMode::RC) {            \
        if (data_format == ENUM::TYPE0A || data_format == ENUM::TYPE0B) {                     \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                  \
                ckernel::sfpu::CALC0<APPROXIMATE>, dst_index, vector_mode);                   \
        } else if (data_format == ENUM::MODE1) {                                              \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                  \
                ckernel::sfpu::CALC1<APPROXIMATE>, dst_index, vector_mode);                   \
        }                                                                                     \
    }

#define SFPU_UNARY_KERNEL(OP)                                                                             \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                                    \
    }                                                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode);                          \
    }

#define SFPU_UNARY_PARAMS_KERNEL(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                         \
    template <bool APPROXIMATE>                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                        \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(sfpu::OP##_init<APPROXIMATE>); \
    }                                                                                              \
    template <bool APPROXIMATE>                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                  \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {                 \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                           \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS);   \
    }

// For the int32 comparison variants
#define SFPU_COMP_INT32_KERNEL(OP, TYPE)                                                                           \
    namespace ckernel {                                                                                            \
    template <bool APPROXIMATE>                                                                                    \
    inline void llk_math_eltwise_unary_sfpu_unary_##OP##_int32(                                                    \
        uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {                                      \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                           \
            ckernel::sfpu::calculate_comp_unary_int<APPROXIMATE, SfpuType::TYPE>, dst_index, vector_mode, param0); \
    }                                                                                                              \
    }

// For the int32 comparison variants with underscore in callback
#define SFPU_COMP_INT32_KERNEL_UNDERSCORE(OP, TYPE)                                                                  \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_unary_##OP##_int32(                                                      \
        uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {                                        \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                             \
            ckernel::sfpu::_calculate_comp_unary_int_<APPROXIMATE, SfpuType::TYPE>, dst_index, vector_mode, param0); \
    }

// For the "normal" comparison ops
#define SFPU_COMP_KERNEL(OP)                                                                   \
    namespace ckernel {                                                                        \
    template <bool APPROXIMATE>                                                                \
    inline void llk_math_eltwise_unary_sfpu_unary_##OP##_init() {                              \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unary_##OP, APPROXIMATE>();                 \
    }                                                                                          \
    template <bool APPROXIMATE>                                                                \
    inline void llk_math_eltwise_unary_sfpu_unary_##OP(                                        \
        uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                       \
            ckernel::sfpu::calculate_unary_##OP<APPROXIMATE>, dst_index, vector_mode, param0); \
    }                                                                                          \
    }

// For ops that need a custom init callback but no extra arguments
#define SFPU_INIT_KERNEL(OP, INIT_CB)                                                                     \
    /* init with callback */                                                                              \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
                                                                                                          \
    /* default compute */                                                                                 \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode);                          \
    }

#define SFPU_INIT_AND_COMPUTE_TWO_TEMPLATE(OP, INIT_CB, FN, T1, T2)                                       \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::FN<APPROXIMATE, T1, T2>, dst_index, vector_mode);                              \
    }

#define SFPU_INIT_LITERAL_KERNEL(OP, INIT_CB, LIT)                                                        \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE, false>, /* base_flag = false*/                     \
            dst_index,                                                                                    \
            vector_mode,                                                                                  \
            LIT);                                                                                         \
    }

// For ops with exactly one extra uint parameter (and no custom init callback)
#define SFPU_UNARY_ONE_PARAM_KERNEL(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                    \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                      \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                           \
    }                                                                                            \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {               \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For ops needing an init with callback and compute with a required param0, RC mode fixed
#define SFPU_INIT_AND_ONE_PARAM_RC_KERNEL(OP, INIT_CB)                                           \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                      \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);       \
    }                                                                                            \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, uint param0) {                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, (int)VectorMode::RC, param0); \
    }

// For ops that need a custom init callback but takes one extra init-parameter
#define SFPU_INIT_ONE_PARAM_KERNEL(OP, INIT_CB, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                            \
    namespace ckernel {                                                                                    \
    template <bool APPROXIMATE>                                                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(EXTRA_ARG_DECL) {                                  \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, EXTRA_ARG_PASS); \
    }                                                                                                      \
    }

// Trig ops with exactly one dst_index argument and RC mode
#define SFPU_TRIG_KERNEL(OP)                                                                                \
    template <bool APPROXIMATE>                                                                             \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                                 \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                                      \
    }                                                                                                       \
    template <bool APPROXIMATE>                                                                             \
    inline void llk_math_eltwise_unary_sfpu_##OP##_op(uint dst_index) {                                     \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                    \
            ckernel::sfpu::calculate_sfpu_trig<SfpuType::OP, APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }

// Inverse-hyperbolic (acosh/asinh): single init + single compute
#define SFPU_INVERSE_HYPERBOLIC_KERNEL(OP, ITER)                                                          \
    /* init */                                                                                            \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(                                      \
            ckernel::sfpu::_init_inverse_hyperbolic_<APPROXIMATE>);                                       \
    }                                                                                                     \
    /* compute */                                                                                         \
    template <bool APPROXIMATE, int ITERATIONS = ITER>                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::_calculate_##OP##_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);          \
    }

// For ops with exactly one dst_index and RC mode
#define SFPU_SIMPLE_OP_KERNEL(OP)                                                        \
    template <bool APPROXIMATE>                                                          \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                              \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                   \
    }                                                                                    \
    template <bool APPROXIMATE>                                                          \
    inline void llk_math_eltwise_unary_sfpu_##OP##_op(uint dst_index) {                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                 \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }

// For compute kernels named ..._op(uint dst_index), with fixed RC mode
#define SFPU_OP_SUFFIX_KERNEL(OP)                                                        \
    template <bool APPROXIMATE>                                                          \
    inline void llk_math_eltwise_unary_sfpu_##OP##_op(uint dst_index) {                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                 \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }

// For erf and erfc which share one compute callback
#define SFPU_ERF_ERFC_KERNEL(NAME, TYPE)                                                                          \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                                     \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>();                                          \
    }                                                                                                             \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index, int param0 /*= 0*/) {                          \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                          \
            ckernel::sfpu::calculate_sfpu_erf_erfc<SfpuType::TYPE, APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }

#define SFPU_TEMPLATE_INIT_KERNEL(OP, INIT_CB)                                                                 \
    namespace ckernel {                                                                                        \
    template <bool APPROXIMATE, bool FAST_APPROX, uint32_t scale = 0x3F800000>                                 \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                                    \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE, FAST_APPROX, scale>); \
    }                                                                                                          \
    }

#define SFPU_TEMPLATE_PARAMS_KERNEL(                                                                            \
    OP,                                                                                                         \
    MODE,                                                                                                       \
    FAST_ARG_DECL,                                                                                              \
    SCALE_ARG_DECL,                                                                                             \
    SKIP_ARG_DECL,                                                                                              \
    IT_ARG_DECL,                                                                                                \
    FAST_ARG_PASS,                                                                                              \
    SCALE_ARG_PASS,                                                                                             \
    SKIP_ARG_PASS,                                                                                              \
    IT_ARG_PASS)                                                                                                \
    namespace ckernel {                                                                                         \
    template <bool APPROXIMATE, bool FAST_APPROX, bool SCALE_EN, bool SKIP_POSITIVE_CHECK, int ITERATIONS>      \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                               \
        uint dst_index,                                                                                         \
        int vector_mode = (int)VectorMode::MODE,                                                                \
        IT_ARG_DECL param0 = ITERATIONS,                                                                        \
        SCALE_ARG_DECL param1 = 0x3F80) {                                                                       \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                        \
            ckernel::sfpu::calculate_##OP<APPROXIMATE, FAST_APPROX, SCALE_EN, ITERATIONS, SKIP_POSITIVE_CHECK>, \
            dst_index,                                                                                          \
            vector_mode,                                                                                        \
            IT_ARG_PASS,                                                                                        \
            SCALE_ARG_PASS);                                                                                    \
    }                                                                                                           \
    }

#define SFPU_INIT_PARAMS_KERNEL(OP, INIT_CB, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                               \
    template <bool APPROXIMATE>                                                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(EXTRA_ARG_DECL) {                                  \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, EXTRA_ARG_PASS); \
    }

// For a simple compute‐only kernel for ops whose sfpu::calculate_<OP> takes two template parameters: <APPROXIMATE,
// LITERAL_ITERATIONS>
#define SFPU_SIMPLE_TWO_PARAM_KERNEL_SUFFIX(OP, FN, ITER)                                                              \
    template <bool APPROXIMATE>                                                                                        \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) {              \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, ITER>, dst_index, vector_mode); \
    }

// For a trivial no‐compute init for identity‐style ops
#define SFPU_IDENTITY_INIT()                                               \
    namespace ckernel {                                                    \
    template <bool APPROXIMATE>                                            \
    inline void llk_math_eltwise_unary_sfpu_identity_init() {              \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(); \
    }                                                                      \
    }

#define SFPU_INIT_KERNEL_NOARG(OP, INIT_CB)                                                    \
    template <bool APPROXIMATE>                                                                \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                    \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(INIT_CB<APPROXIMATE>); \
    }

#define SFPU_DIM_SWITCH_KERNEL(OP, ENUM, CALC0, MODE0, CALC1, MODE1)                   \
    template <bool APPROXIMATE>                                                        \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, ENUM sum_int_dim) {   \
        if (sum_int_dim == ENUM::SUM_COL) {                                            \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                           \
                ckernel::sfpu::CALC0<APPROXIMATE>, dst_index, (int)VectorMode::MODE0); \
        } else if (sum_int_dim == ENUM::SUM_ROW) {                                     \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                           \
                ckernel::sfpu::CALC1<APPROXIMATE>, dst_index, (int)VectorMode::MODE1); \
        }                                                                              \
    }

#define SFPU_ONE_PARAM_CONST_ITERS_KERNEL(OP, ITERS)                                              \
    template <bool APPROXIMATE>                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                 \
        uint dst_index, uint dst_offset, int iterations, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                          \
            ckernel::sfpu::OP<APPROXIMATE, ITERS>, dst_index, vector_mode, dst_offset);           \
    }

// For isinf / isposinf / isneginf / isnan / isfinite - they all share the same calculate_sfpu_isinf_isnan<...> callback
#define SFPU_ISINF_ISNAN_KERNEL(NAME, TYPE)                                                                          \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                                        \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>();                                             \
    }                                                                                                                \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index) {                                                 \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                             \
            ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::TYPE, APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }

// For logical_not_unary (float + int32 variants) which share one init and two custom callbacks
#define SFPU_LOGICAL_NOT_NOTI_KERNEL(NAME, VT0, ET0, VT1, ET1)                          \
    template <bool APPROXIMATE>                                                         \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                           \
        llk_math_eltwise_unary_sfpu_init<SfpuType::NAME, APPROXIMATE>();                \
    }                                                                                   \
    template <bool APPROXIMATE>                                                         \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_op(uint dst_index) {               \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                \
            ckernel::sfpu::calculate_##NAME<VT0, ET0>, dst_index, (int)VectorMode::RC); \
    }                                                                                   \
    template <bool APPROXIMATE>                                                         \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_op_int32(uint dst_index) {         \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                \
            ckernel::sfpu::calculate_##NAME<VT1, ET1>, dst_index, (int)VectorMode::RC); \
    }

// For a rand kernel needs both a custom init(seed) and a fixed-RC compute(from,scale)
#define SFPU_RAND_KERNEL(OP, INIT_CB)                                                                 \
    template <bool APPROXIMATE>                                                                       \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(uint32_t seed = 0) {                          \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(INIT_CB<APPROXIMATE>, seed);  \
    }                                                                                                 \
    template <bool APPROXIMATE>                                                                       \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint32_t dst_index, uint32_t from, uint32_t scale) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                              \
            ckernel::sfpu::OP<APPROXIMATE>, dst_index, (int)VectorMode::RC, from, scale);             \
    }

// For ops with two template parameters <APPROXIMATE, is_fp32_dest_acc_en> and a fixed third template arg = 8
#define SFPU_TEMPLATE_TWO_PARAM_KERNEL(OP, INIT_CB)                                                       \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
    template <bool APPROXIMATE, bool is_fp32_dest_acc_en>                                                 \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode);  \
    }

// For cases where the OP name does NOT match the SfpuType and/or callback
#define SFPU_UNARY_OP_INIT(OP, TYPE)                                     \
    template <bool APPROXIMATE>                                          \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {              \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>(); \
    }

// For compute functions with custom callback (param0=0 default)
#define SFPU_UNARY_OP_COMPUTE(OP, CB)                                                \
    template <bool APPROXIMATE>                                                      \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, uint param0 = 0) {  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                             \
            ckernel::sfpu::CB<APPROXIMATE>, dst_index, (int)VectorMode::RC, param0); \
    }

#define SFPU_RELU_ALIAS()                                                \
    template <bool APPROXIMATE>                                          \
    inline void llk_math_eltwise_unary_sfpu_relu(uint dst_index) {       \
        llk_math_eltwise_unary_sfpu_relu_min<APPROXIMATE>(dst_index, 0); \
    }

// For a kernel with exactly two extra uint parameters (init and compute)
#define SFPU_UNARY_TWO_PARAM_KERNEL(OP, MODE, INIT_CB, EXTRA1_DECL, EXTRA2_DECL, EXTRA1_PASS, EXTRA2_PASS)           \
    /* init takes two uints */                                                                                       \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(EXTRA1_DECL, EXTRA2_DECL) {                                  \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, EXTRA1_PASS, EXTRA2_PASS); \
    }                                                                                                                \
    /* compute takes dst_index + those same two uints + RC mode */                                                   \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                                    \
        uint dst_index, EXTRA1_DECL, EXTRA2_DECL, int vector_mode = (int)VectorMode::MODE) {                         \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                             \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA1_PASS, EXTRA2_PASS);           \
    }

#define SFPU_RSUB_KERNEL()                                                                             \
    /* init */                                                                                         \
    template <bool APPROXIMATE>                                                                        \
    inline void llk_math_eltwise_unary_sfpu_rsub_init() {                                              \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::rsub_init<APPROXIMATE>); \
    }                                                                                                  \
    /* compute(dst, param0=0) */                                                                       \
    template <bool APPROXIMATE>                                                                        \
    inline void llk_math_eltwise_unary_sfpu_rsub(uint dst_index, uint param0 = 0) {                    \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                               \
            ckernel::sfpu::calculate_rsub<APPROXIMATE, 8>, dst_index, (int)VectorMode::RC, param0);    \
    }

// For ops where the compute callback takes an extra template parameter (e.g., bool flag)
#define SFPU_UNARY_PARAMS_KERNEL_WITH_EXTRA_TEMPLATE_ARG(                                          \
    OP, MODE, CALC_CB, EXTRA_TEMPLATE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                             \
    template <bool APPROXIMATE>                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                        \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(sfpu::OP##_init<APPROXIMATE>); \
    }                                                                                              \
    template <bool APPROXIMATE>                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                  \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {                 \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                           \
            CALC_CB<APPROXIMATE, EXTRA_TEMPLATE>, dst_index, vector_mode, EXTRA_ARG_PASS);         \
    }

// For ops where the compute function takes an extra argument (with default) as the third parameter, after vector_mode
#define SFPU_UNARY_ONE_PARAM_KERNEL_MODE_SECOND_DEFAULT(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS, EXTRA_ARG_DEFAULT) \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                                          \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                                               \
    }                                                                                                                \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                                    \
        uint dst_index, int vector_mode = (int)VectorMode::MODE, EXTRA_ARG_DECL = EXTRA_ARG_DEFAULT) {               \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                             \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS);                     \
    }

// For ops with extra args, but default init (no callback)
#define SFPU_UNARY_PARAMS_KERNEL_NO_INITCB(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)             \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                      \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                           \
    }                                                                                            \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {               \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For kernels that only have non-approximate (false) implementation
#define SFPU_UNARY_PARAMS_KERNEL_APPROX_FALSE(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)    \
    template <bool APPROXIMATE>                                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, false>(sfpu::OP##_init<false>);     \
    }                                                                                      \
    template <bool APPROXIMATE>                                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                          \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {         \
        llk_math_eltwise_unary_sfpu_params<false>(                                         \
            ckernel::sfpu::calculate_##OP<false>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For the compare with zero ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL(NAME, TYPE, LITERAL)                                                                       \
    /* init */                                                                                                      \
    template <bool APPROXIMATE>                                                                                     \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                                       \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>();                                            \
    }                                                                                                               \
                                                                                                                    \
    /* fp variant: pass literal LITERAL as extra arg */                                                             \
    template <bool APPROXIMATE>                                                                                     \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index, int vector_mode = (int)VectorMode::RC) {         \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                            \
            ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::TYPE>, dst_index, vector_mode, LITERAL);           \
    }                                                                                                               \
                                                                                                                    \
    /* int32 variant: no literal */                                                                                 \
    template <bool APPROXIMATE>                                                                                     \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_int32(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                            \
            ckernel::sfpu::calculate_comp_int<APPROXIMATE, SfpuType::TYPE>, dst_index, vector_mode);                \
    }

// Generates init, float, and int32 kernels for unary_min/unary_max, selecting <is_max, APPROXIMATE> on the shared
// calculate_unary_max_min/calculate_unary_max_min_int32
#define SFPU_UNARY_MAXMIN_KERNEL(NAME, IS_MAX, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                                    \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                                     \
        llk_math_eltwise_unary_sfpu_init<SfpuType::NAME, APPROXIMATE>();                                          \
    }                                                                                                             \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME(                                                               \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::RC) {                                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                          \
            ckernel::sfpu::calculate_unary_max_min<IS_MAX, APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }                                                                                                             \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_int32(                                                       \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::RC) {                                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                          \
            ckernel::sfpu::calculate_unary_max_min_int32<IS_MAX, APPROXIMATE>,                                    \
            dst_index,                                                                                            \
            vector_mode,                                                                                          \
            EXTRA_ARG_PASS);                                                                                      \
    }
