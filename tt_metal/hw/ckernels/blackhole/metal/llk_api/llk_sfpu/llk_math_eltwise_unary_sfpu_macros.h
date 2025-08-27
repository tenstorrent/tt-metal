// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// For ops that require only the init function
#define SFPU_UNARY_KERNEL_INIT(OP, APPROXIMATE) llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();

// For ops that need a custom init callback
#define SFPU_INIT_KERNEL_CALL(OP, INIT_CB, APPROXIMATE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>)

// For ops that need a custom init callback with fast_approx template parameter
#define SFPU_INIT_KERNEL_CALL_FAST_APPROX(OP, INIT_CB, APPROXIMATE, FAST_APPROX) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE, FAST_APPROX>)

// For ops that need a custom init callback but takes one extra init-parameter
#define SFPU_ONE_PARAM_KERNEL_INIT(OP, INIT_CB, APPROXIMATE, PARAM0) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, PARAM0);

// For ops that need a custom init callback and take two extra init-parameters
#define SFPU_TWO_PARAM_KERNEL_INIT(OP, INIT_CB, APPROXIMATE, PARAM0, PARAM1) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, PARAM0, PARAM1)

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
#define SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(FN, MODE, APPROXIMATE, DST_IDX, PARAM0, PARAM1) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                      \
        ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0, PARAM1);

// For ops with multiple template parameters and one runtime parameter (e.g., scale)
#define SFPU_TEMPLATE_PARAMS_KERNEL(                                                                                  \
    OP,                                                                                                               \
    APPROXIMATE,                                                                                                      \
    FAST_APPROX,                                                                                                      \
    SCALE_EN,                                                                                                         \
    SKIP_POSITIVE_CHECK,                                                                                              \
    ITERATIONS,                                                                                                       \
    IS_FP32_DEST_ACC_EN,                                                                                              \
    DST_IDX,                                                                                                          \
    VECTOR_MODE,                                                                                                      \
    SCALE)                                                                                                            \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                                                \
        ckernel::sfpu::                                                                                               \
            calculate_##OP<APPROXIMATE, FAST_APPROX, SCALE_EN, ITERATIONS, SKIP_POSITIVE_CHECK, IS_FP32_DEST_ACC_EN>, \
        DST_IDX,                                                                                                      \
        VECTOR_MODE,                                                                                                  \
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
#define SFPU_TWO_PARAM_KERNEL_ONE_RUNTIME(FN, APPROXIMATE, T2, DST_IDX, VECTOR_MODE, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, T2>, DST_IDX, VECTOR_MODE, PARAM0)

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

// Generalized macro for compare-with-zero ops for any type (e.g., int, uint16)
#define SFPU_ZERO_KERNEL_TYPE(TYPE_SUFFIX, OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                     \
        ckernel::sfpu::calculate_comp_##TYPE_SUFFIX<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE);

// For log1p op that needs both APPROXIMATE and FAST_APPROX template parameters
#define SFPU_UNARY_NO_PARAM_KERNEL_LOG1P(OP, MODE, APPROXIMATE, FAST_APPROX, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                                \
        ckernel::sfpu::calculate_##OP<APPROXIMATE, FAST_APPROX>, DST_IDX, (int)VectorMode::MODE)

// Generalized variadic macro for unary kernels with any number of extra runtime parameters
#define SFPU_UNARY_KERNEL_VARIADIC(OP, MODE, APPROXIMATE, DST_IDX, ...) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                  \
        ckernel::sfpu::calculate_##OP<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, ##__VA_ARGS__)
