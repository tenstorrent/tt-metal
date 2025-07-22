// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_bitwise_and.h"
#include "ckernel_sfpu_bitwise_or.h"
#include "ckernel_sfpu_bitwise_not.h"
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

// For ops where init takes multiple template parameters (e.g., approximate, fast_approx, scale).
#define SFPU_TEMPLATE_INIT_KERNEL(OP, INIT_CB, APPROX, FAST_APPROX, SCALE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX>(INIT_CB<APPROX, FAST_APPROX, SCALE>)

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

// For the compare with zero ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL(OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(   \
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, 8);

// For the compare with int32 zero ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL_INT32(OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(         \
        ckernel::sfpu::calculate_comp_int<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE);
