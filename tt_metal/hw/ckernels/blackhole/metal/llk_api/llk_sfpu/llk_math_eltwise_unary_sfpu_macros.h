// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_bitwise_or.h"
#include "ckernel_sfpu_bitwise_xor.h"
#include "ckernel_sfpu_comp.h"
#include "ckernel_sfpu_dropout.h"
#include "ckernel_sfpu_elu.h"
#include "ckernel_sfpu_erf_erfc.h"
#include "ckernel_sfpu_erfinv.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_fill.h"
#include "ckernel_sfpu_fmod.h"
#include "ckernel_sfpu_gelu.h"
#include "ckernel_sfpu_i0.h"
#include "ckernel_sfpu_i1.h"
#include "ckernel_sfpu_identity.h"
#include "ckernel_sfpu_int_sum.h"
#include "ckernel_sfpu_isinf_isnan.h"
#include "ckernel_sfpu_left_shift.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_logical_not_noti.h"
#include "ckernel_sfpu_negative.h"
#include "ckernel_sfpu_prelu.h"
#include "ckernel_sfpu_rand.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_relu.h"
#include "ckernel_sfpu_remainder.h"
#include "ckernel_sfpu_right_shift.h"
#include "ckernel_reverseops.h"
#include "ckernel_sfpu_softplus.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_trigonometry.h"
#include "ckernel_sfpu_unary_comp.h"

// For ops that require only the init function
#define SFPU_UNARY_KERNEL_INIT(OP, APPROXIMATE) llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();

// For ops that need a custom init callback
#define SFPU_INIT_KERNEL_CALL(OP, INIT_CB, APPROXIMATE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>)

// For ops that need a custom init callback but takes one extra init-parameter
#define SFPU_ONE_PARAM_KERNEL_INIT(OP, INIT_CB, APPROXIMATE, EXTRA_ARG) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, EXTRA_ARG);

// For ops that need a custom init callback and take two extra init-parameters
#define SFPU_TWO_PARAM_KERNEL_INIT(OP, INIT_CB, APPROXIMATE, ARG0, ARG1) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, ARG0, ARG1)

// For ops that require only an init function with a custom SfpuType and init callback
#define SFPU_INIT_ONLY_WITH_TYPE(NAME, TYPE, INIT_CB)                                        \
    template <bool APPROXIMATE>                                                              \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>(INIT_CB<APPROXIMATE>); \
    }

// For ops where init takes multiple template parameters (e.g., approximate, fast_approx, scale).
#define SFPU_TEMPLATE_INIT_KERNEL(OP, INIT_CB, APPROX, FAST_APPROX, SCALE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX>(INIT_CB<APPROX, FAST_APPROX, SCALE>)

// For ops that require only the compute function
#define SFPU_UNARY_KERNEL_NO_INIT(OP)                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode);                          \
    }

// For ops with extra arguments that require a custom compute callback instead of the default calculate_<OP>
#define SFPU_UNARY_PARAMS_KERNEL_WITH_CUSTOM_CALC(OP, MODE, CALC_CB, EXTRA_ARG_DECL, EXTRA_ARG_PASS) \
    template <bool APPROXIMATE>                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                    \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {                   \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                           \
            CALC_CB<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS);                           \
    }

// For unary kernels with multiple extra arguments and a custom compute callback
#define SFPU_UNARY_PARAMS_KERNEL_CUSTOM_ARGS(OP, MODE, CALC_CB, EXTRA_DECLS, EXTRA_PASS)                           \
    template <bool APPROXIMATE>                                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                                  \
        uint dst_index, EXTRA_DECLS, int vector_mode = (int)VectorMode::MODE) {                                    \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(CALC_CB<APPROXIMATE>, dst_index, vector_mode, EXTRA_PASS); \
    }

// For unary ops that also need an _int32 variant
#define SFPU_UNARY_INT32_KERNEL(OP)                                                                               \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##OP##_int32(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                        \
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

// For ops with two implementations that select compute callback based on an enum argument
#define SFPU_DIM_DUALTYPE_SWITCH_KERNEL(OP, ENUM, TYPE0A, TYPE0B, CALC0, MODE0, CALC1, MODE1) \
    template <bool APPROXIMATE>                                                               \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                             \
        uint dst_index, ENUM data_format, int vector_mode = (int)VectorMode::RC) {            \
        if (data_format == ENUM::TYPE0A || data_format == ENUM::TYPE0B) {                     \
            _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                \
                ckernel::sfpu::CALC0<APPROXIMATE>, dst_index, vector_mode);                   \
        } else if (data_format == ENUM::MODE1) {                                              \
            _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                \
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
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                \
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
#define SFPU_COMP_INT32_KERNEL(OP, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                 \
        ckernel::sfpu::calculate_comp_unary_int<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, PARAM0);

// For the int32 comparison variants with underscore in callback
#define SFPU_COMP_INT32_KERNEL_UNDERSCORE(OP, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                            \
        ckernel::sfpu::_calculate_comp_unary_int_<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, PARAM0);

// For the int32 comparison variants with underscores in callback(ge, le)
#define SFPU_COMP_KERNEL(OP, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(           \
        ckernel::sfpu::_calculate_comp_unary_<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, PARAM0);

#define SFPU_TOPK_LOCAL_SORT_KERNEL(OP)                                      \
    template <bool APPROXIMATE>                                              \
    inline void llk_math_eltwise_unary_sfpu_##OP(                            \
        uint dst_index,                                                      \
        int idir,                                                            \
        int i_end_phase,                                                     \
        int i_start_phase,                                                   \
        int i_end_step,                                                      \
        int i_start_step,                                                    \
        int vector_mode = (int)VectorMode::RC_custom) {                      \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                   \
            ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROXIMATE>, \
            dst_index,                                                       \
            vector_mode,                                                     \
            idir,                                                            \
            i_end_phase,                                                     \
            i_start_phase,                                                   \
            i_end_step,                                                      \
            i_start_step);                                                   \
    }

// Macro for topk_merge which expects multiple arguments and a custom compute callback
#define SFPU_TOPK_MERGE_KERNEL(OP)                                                                              \
    template <bool APPROXIMATE, bool idir = false>                                                              \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                               \
        uint dst_index, int m_iter, int k, int vector_mode = (int)VectorMode::RC_custom) {                      \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                      \
            ckernel::sfpu::calculate_bitonic_topk_merge<APPROXIMATE, idir>, dst_index, vector_mode, m_iter, k); \
    }

#define SFPU_TOPK_REBUILD_KERNEL(OP)                                    \
    template <bool APPROXIMATE>                                         \
    inline void llk_math_eltwise_unary_sfpu_##OP(                       \
        uint dst_index,                                                 \
        bool idir,                                                      \
        int m_iter,                                                     \
        int k,                                                          \
        int logk,                                                       \
        int skip_second,                                                \
        int vector_mode = (int)VectorMode::RC_custom) {                 \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(              \
            ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROXIMATE>, \
            dst_index,                                                  \
            vector_mode,                                                \
            idir,                                                       \
            m_iter,                                                     \
            k,                                                          \
            logk,                                                       \
            skip_second);                                               \
    }

// For ops where init callback is custom and compute callback uses an extra template parameter
#define SFPU_INIT_AND_UNARY_PARAMS_KERNEL_WITH_EXTRA_TEMPLATE_ARG(                           \
    OP, TYPE, INIT_CB, MODE, CALC_CB, EXTRA_TEMPLATE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)        \
    template <bool APPROXIMATE>                                                              \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                  \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>(INIT_CB<APPROXIMATE>); \
    }                                                                                        \
    template <bool APPROXIMATE>                                                              \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                            \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {           \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                   \
            CALC_CB<APPROXIMATE, EXTRA_TEMPLATE>, dst_index, vector_mode, EXTRA_ARG_PASS);   \
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
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode);                          \
    }

#define SFPU_INIT_AND_COMPUTE_TWO_TEMPLATE(OP, INIT_CB, FN, T1, T2)                                       \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                \
            ckernel::sfpu::FN<APPROXIMATE, T1, T2>, dst_index, vector_mode);                              \
    }

#define SFPU_INIT_LITERAL_KERNEL(OP, INIT_CB, LIT)                                                        \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                \
            ckernel::sfpu::calculate_##OP<APPROXIMATE, false>, /* base_flag = false*/                     \
            dst_index,                                                                                    \
            vector_mode,                                                                                  \
            LIT);                                                                                         \
    }

// For ops with exactly one extra uint parameter (and no custom init callback)
#define SFPU_UNARY_ONE_PARAM_KERNEL(OP, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                      \
        ckernel::sfpu::calculate_##OP<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For ops where the compute functor is not "calculate_<OP>", but an arbitrary function name (e.g., relu_min, relu_max,
// etc.),and which take one extra runtime uint parameter
#define SFPU_UNARY_ONE_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                         \
        ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For ops with exactly two extra uint parameters (and no custom init callback)
#define SFPU_UNARY_TWO_PARAM_KERNEL(OP, MODE, APPROXIMATE, DST_IDX, PARAM0, PARAM1) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                              \
        ckernel::sfpu::calculate_##OP<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0, PARAM1)

// For ops with exactly three extra uint parameters (and no custom init callback)
#define SFPU_UNARY_THREE_PARAM_KERNEL(OP, MODE, APPROXIMATE, DST_IDX, PARAM0, PARAM1, PARAM2) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                        \
        ckernel::sfpu::calculate_##OP<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0, PARAM1, PARAM2)

// For ops without extra uint parameter
#define SFPU_UNARY_NO_PARAM_KERNEL(OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(             \
        ckernel::sfpu::calculate_##OP<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE)

// For ops without extra uint parameter with type
#define SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE(OP, TYPE, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                             \
        ckernel::sfpu::calculate_##OP<SfpuType::TYPE, APPROXIMATE>, DST_IDX, (int)VectorMode::MODE)

// For unary ops with one extra uint parameter AND an additional template param (ITERATIONS)
#define SFPU_UNARY_ONE_PARAM_KERNEL_ITER(FN, MODE, APPROXIMATE, ITER, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                 \
        ckernel::sfpu::FN<APPROXIMATE, ITER>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For kernels with two template params and an extra integer template param (e.g., 8).
#define SFPU_THREE_TEMPLATE_PARAM_KERNEL(OP, APPROXIMATE, DST_ACCUM_MODE, DEFAULT_INT, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                       \
        ckernel::sfpu::calculate_##OP<APPROXIMATE, DST_ACCUM_MODE, DEFAULT_INT>, DST_IDX, VECTOR_MODE)

// For ops where compute takes extra args
#define SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(OP, MODE, APPROXIMATE, DST_IDX, ARG0, ARG1) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                  \
        ckernel::sfpu::calculate_##OP<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, ARG0, ARG1);

// For ops with multiple template parameters and one runtime parameter (e.g., scale)
#define SFPU_TEMPLATE_PARAMS_KERNEL(                                                                        \
    OP, APPROXIMATE, FAST_APPROX, SCALE_EN, SKIP_POSITIVE_CHECK, ITERATIONS, DST_IDX, VECTOR_MODE, SCALE)   \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                      \
        ckernel::sfpu::calculate_##OP<APPROXIMATE, FAST_APPROX, SCALE_EN, ITERATIONS, SKIP_POSITIVE_CHECK>, \
        DST_IDX,                                                                                            \
        VECTOR_MODE,                                                                                        \
        SCALE)

// For kernels with one template parameter and one extra runtime argument.
#define SFPU_UNARY_ONE_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                         \
        ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For kernels whose functor takes one template parameter (e.g., <APPROXIMATE>)
#define SFPU_ONE_PARAM_KERNEL(FN, APPROXIMATE, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, VECTOR_MODE)

// For kernels whose functor takes two template parameters (e.g., <APPROXIMATE, ITER/USE_FP32>)
#define SFPU_TWO_PARAM_KERNEL(FN, APPROXIMATE, T2, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, T2>, DST_IDX, VECTOR_MODE)

// For kernels whose functor takes two template parameters (e.g., <APPROXIMATE, ITER>) and one extra runtime param.
#define SFPU_TWO_PARAM_KERNEL_ONE_RUNTIME(FN, APPROXIMATE, T2, DST_IDX, VECTOR_MODE, EXTRA) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, T2>, DST_IDX, VECTOR_MODE, EXTRA)

// For kernels whose functor takes three template parameters (e.g., <APPROXIMATE, ITER, USE_FP32>)
#define SFPU_THREE_PARAM_KERNEL_ITER_FIRST(FN, APPROXIMATE, ITER, USE_FP32, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                            \
        ckernel::sfpu::FN<APPROXIMATE, ITER, USE_FP32>, DST_IDX, VECTOR_MODE)

// For functors with <APPROXIMATE, USE_FP32, ITER>
#define SFPU_THREE_PARAM_KERNEL_USEFP32_FIRST(FN, APPROXIMATE, USE_FP32, ITER, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                               \
        ckernel::sfpu::FN<APPROXIMATE, USE_FP32, ITER>, DST_IDX, VECTOR_MODE)

// For kernels which takes two extra template parameters (e.g., <V, T>)
#define SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS(OP, APPROXIMATE, V, T, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                        \
        ckernel::sfpu::calculate_##OP<V, T>, DST_IDX, static_cast<int>(VectorMode::RC));

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
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                           \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA1_PASS, EXTRA2_PASS);           \
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
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                           \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS);                     \
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
        _llk_math_eltwise_unary_sfpu_params_<false>(                                       \
            ckernel::sfpu::calculate_##OP<false>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }

// For the compare with zero ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL(OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(   \
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, 8);

// For the compare with int32 zero ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL_INT32(OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(         \
        ckernel::sfpu::calculate_comp_int<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE);

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
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                        \
            ckernel::sfpu::calculate_unary_max_min<IS_MAX, APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }                                                                                                             \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_int32(                                                       \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::RC) {                                  \
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                        \
            ckernel::sfpu::calculate_unary_max_min_int32<IS_MAX, APPROXIMATE>,                                    \
            dst_index,                                                                                            \
            vector_mode,                                                                                          \
            EXTRA_ARG_PASS);                                                                                      \
    }
