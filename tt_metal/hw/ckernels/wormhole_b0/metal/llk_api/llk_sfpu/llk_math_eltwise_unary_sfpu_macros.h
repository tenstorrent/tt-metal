// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// For ops that require only the init function
#define SFPU_UNARY_KERNEL_INIT(OP, APPROX_MODE) llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX_MODE>();

// For ops that need a custom init callback
#define SFPU_INIT_KERNEL_CALL(OP, INIT_CB, APPROX_MODE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX_MODE>(INIT_CB<APPROX_MODE>)

// For ops that need a custom init callback but takes one extra init-parameter
#define SFPU_ONE_PARAM_KERNEL_INIT(OP, INIT_CB, APPROX_MODE, PARAM0) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX_MODE>(INIT_CB<APPROX_MODE>, PARAM0);

// For ops that need a custom init callback with two template parameters (e.g., APPROX_MODE and legacy_compat or
// fast_approx)
#define SFPU_TWO_TEMPLATE_PARAM_INIT(OP, INIT_CB, APPROX_MODE, PARAM0) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX_MODE>(INIT_CB<APPROX_MODE, PARAM0>)

// For ops that need a custom init callback with three template parameters (e.g., APPROX_MODE, DST_ACCUM and
// legacy_compat)
#define SFPU_THREE_TEMPLATE_PARAM_INIT(OP, INIT_CB, APPROX_MODE, PARAM0, PARAM1) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX_MODE>(INIT_CB<APPROX_MODE, PARAM0, PARAM1>)

// For ops that need a custom init callback and take two extra init-parameters
#define SFPU_TWO_PARAM_KERNEL_INIT(OP, INIT_CB, APPROX_MODE, PARAM0, PARAM1) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX_MODE>(INIT_CB<APPROX_MODE>, PARAM0, PARAM1)

// For ops where init takes multiple template parameters (e.g., approximate, fast_approx, scale).
#define SFPU_TEMPLATE_INIT_KERNEL(OP, INIT_CB, APPROX, SCALE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX>(INIT_CB<APPROX, SCALE>)

// For the int32 comparison variants
#define SFPU_COMP_INT32_KERNEL(OP, MODE, APPROX_MODE, ITERATIONS, DST_IDX, PARAM0)      \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                  \
        ckernel::sfpu::calculate_comp_unary_int<APPROX_MODE, SfpuType::OP, ITERATIONS>, \
        DST_IDX,                                                                        \
        (int)VectorMode::MODE,                                                          \
        PARAM0);

// For the int32 comparison variants with underscore in callback
#define SFPU_COMP_INT32_KERNEL_UNDERSCORE(OP, MODE, APPROX_MODE, ITERATIONS, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                        \
        ckernel::sfpu::_calculate_comp_unary_int_<APPROX_MODE, SfpuType::OP, ITERATIONS>,     \
        DST_IDX,                                                                              \
        (int)VectorMode::MODE,                                                                \
        PARAM0);

// For ops where the compute functor is not "calculate_<OP>", but an arbitrary function name (e.g., relu_min, relu_max,
// etc.),and which take one extra runtime uint parameter
#define SFPU_UNARY_ONE_PARAM_KERNEL_FN(FN, MODE, APPROX_MODE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                         \
        ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For unary ops with one extra uint parameter AND an additional template param (ITERATIONS)
#define SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(FN, MODE, APPROX_MODE, EXTRA_PARAM, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                               \
        ckernel::sfpu::FN<APPROX_MODE, EXTRA_PARAM>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For unary ops with one extra uint parameter AND additional template params (DATA_FORMAT, ITERATIONS)
#define SFPU_UNARY_ONE_PARAM_KERNEL_DATA_FORMAT_EXTRA_PARAM(                                                        \
    FN, MODE, APPROX_MODE, DATA_FORMAT, EXTRA_PARAM, DST_IDX, PARAM0)                                               \
    static_assert(                                                                                                  \
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16, \
        "Unsupported data format. Supported: Int32, UInt32, UInt16");                                               \
    constexpr InstrModLoadStore _INSTRUCTION_MODE =                                                                 \
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;                   \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                                              \
        ckernel::sfpu::FN<APPROX_MODE, _INSTRUCTION_MODE, EXTRA_PARAM>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For ops with exactly two extra uint parameters (and no custom init callback)
#define SFPU_UNARY_TWO_PARAM_KERNEL_FN(FN, MODE, APPROX_MODE, DST_IDX, PARAM0, PARAM1) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                 \
        ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, (int)VectorMode::MODE, PARAM0, PARAM1)

// For ops with exactly three extra uint parameters (and no custom init callback)
#define SFPU_UNARY_THREE_PARAM_KERNEL_FN(FN, MODE, APPROX_MODE, DST_IDX, PARAM0, PARAM1, PARAM2) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                           \
        ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, (int)VectorMode::MODE, PARAM0, PARAM1, PARAM2)

// For ops without extra uint parameter
#define SFPU_UNARY_NO_PARAM_KERNEL_FN(FN, MODE, APPROX_MODE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, (int)VectorMode::MODE)

// For ops without extra uint parameter with ITERATIONS
#define SFPU_UNARY_NO_PARAM_KERNEL_FN_ITERATIONS(FN, MODE, APPROX_MODE, DST_IDX, ITERATIONS) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                       \
        ckernel::sfpu::FN<APPROX_MODE, ITERATIONS>, DST_IDX, (int)VectorMode::MODE)

// For ops without extra uint parameters with type and iteration
#define SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(OP, TYPE, MODE, APPROX_MODE, DST_IDX, ITERATIONS) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                                        \
        ckernel::sfpu::OP<SfpuType::TYPE, APPROX_MODE, ITERATIONS>, DST_IDX, (int)VectorMode::MODE)

// For ops where compute takes extra args
#define SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(FN, MODE, APPROX_MODE, DST_IDX, PARAM0, PARAM1) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                      \
        ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, (int)VectorMode::MODE, PARAM0, PARAM1);

// For ops with multiple template parameters and one runtime parameter (e.g., scale)
#define SFPU_TEMPLATE_PARAMS_KERNEL_FN(                \
    FN,                                                \
    APPROX_MODE,                                       \
    FAST_APPROX,                                       \
    IS_FP32_DEST_ACC_EN,                               \
    SCALE_EN,                                          \
    SKIP_POSITIVE_CHECK,                               \
    CLAMP_NEGATIVE,                                    \
    ITERATIONS,                                        \
    DST_IDX,                                           \
    VECTOR_MODE,                                       \
    SCALE)                                             \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>( \
        ckernel::sfpu::FN<                             \
            APPROX_MODE,                               \
            FAST_APPROX,                               \
            IS_FP32_DEST_ACC_EN,                       \
            SCALE_EN,                                  \
            ITERATIONS,                                \
            SKIP_POSITIVE_CHECK,                       \
            CLAMP_NEGATIVE>,                           \
        DST_IDX,                                       \
        VECTOR_MODE,                                   \
        SCALE)

// For kernels with one template parameter and one extra runtime argument.
#define SFPU_UNARY_ONE_PARAM_KERNEL_FN(FN, MODE, APPROX_MODE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                         \
        ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, (int)VectorMode::MODE, PARAM0)

#define SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(FN, MODE, APPROX_MODE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                               \
        ckernel::sfpu::FN<sfpi::vFloat, APPROX_MODE, 8, uint32_t>, DST_IDX, (int)VectorMode::MODE, PARAM0)

#define SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT(FN, MODE, APPROX_MODE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                             \
        ckernel::sfpu::FN<sfpi::vInt, APPROX_MODE, 8, uint32_t>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// For kernels whose functor takes one template parameter (e.g., <APPROX_MODE>)
#define SFPU_ONE_PARAM_KERNEL(FN, APPROX_MODE, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(ckernel::sfpu::FN<APPROX_MODE>, DST_IDX, VECTOR_MODE)

// For kernels whose functor takes two template parameters (e.g., <APPROX_MODE, ITER/USE_FP32>)
#define SFPU_TWO_PARAM_KERNEL(FN, APPROX_MODE, T2, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(ckernel::sfpu::FN<APPROX_MODE, T2>, DST_IDX, VECTOR_MODE)

// For kernels whose functor takes two template parameters (e.g., <APPROX_MODE, ITER>) and one extra runtime param.
#define SFPU_TWO_PARAM_KERNEL_ONE_RUNTIME(FN, APPROX_MODE, T2, DST_IDX, VECTOR_MODE, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(ckernel::sfpu::FN<APPROX_MODE, T2>, DST_IDX, VECTOR_MODE, PARAM0)

// For kernels whose functor takes three template parameters (e.g., <APPROX_MODE, ITER, USE_FP32>)
#define SFPU_THREE_PARAM_KERNEL_ITER_FIRST(FN, APPROX_MODE, ITER, FP32, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(ckernel::sfpu::FN<APPROX_MODE, ITER, FP32>, DST_IDX, VECTOR_MODE)

// For functors with <APPROX_MODE, USE_FP32, ITER>
#define SFPU_THREE_PARAM_KERNEL_FP32_FIRST(FN, APPROX_MODE, FP32, ITER, DST_IDX, VECTOR_MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(ckernel::sfpu::FN<APPROX_MODE, FP32, ITER>, DST_IDX, VECTOR_MODE)

// For unary ops with four template parameters (APPROX_MODE, ITER, fp32_dest_acc_en, legacy_compat)
#define SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN(FN, APPROX_MODE, ITER, FP32, LEGACY_COMPAT, DST_IDX, MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                                  \
        ckernel::sfpu::FN<APPROX_MODE, ITER, FP32, LEGACY_COMPAT>, DST_IDX, (int)VectorMode::MODE)

// For unary ops with four template parameters (APPROX_MODE, fp32_dest_acc_en, FP32_FIRST, legacy_compat)
#define SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN(FN, APPROX_MODE, FP32, ITER, LEGACY_COMPAT, DST_IDX, MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                                  \
        ckernel::sfpu::FN<APPROX_MODE, FP32, ITER, LEGACY_COMPAT>, DST_IDX, MODE)

// For unary ops with five template parameters (APPROX_MODE, ITER, fp32_dest_acc_en, FAST_APPROX, legacy_compat)
#define SFPU_FIVE_PARAM_KERNEL_ITER_FIRST_FN(FN, APPROX_MODE, ITER, FP32, FAST_APPROX, LEGACY_COMPAT, DST_IDX, MODE) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                                               \
        ckernel::sfpu::FN<APPROX_MODE, ITER, FP32, FAST_APPROX, LEGACY_COMPAT>, DST_IDX, (int)VectorMode::MODE)

// For kernels whose functor takes three template parameters (e.g., <APPROX_MODE, DATA_FORMAT, ITERATIONS>).
#define SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN(FN, APPROX_MODE, DATA_FORMAT, ITERATIONS, DST_IDX, MODE)           \
    static_assert(                                                                                                  \
        DATA_FORMAT == DataFormat::Float32 || DATA_FORMAT == DataFormat::Float16_b ||                               \
            DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 ||                                \
            DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::Bfp8_b,                                 \
        "Unsupported data format. Supported data formats are: Float32, Float16_b, Int32, UInt32, UInt16, Bfp8_b."); \
    constexpr InstrModLoadStore INSTRUCTION_MODE =                                                                  \
        (DATA_FORMAT == DataFormat::Float32 || DATA_FORMAT == DataFormat::Float16_b ||                              \
         DATA_FORMAT == DataFormat::Bfp8_b)                                                                         \
            ? InstrModLoadStore::DEFAULT                                                                            \
        : (DATA_FORMAT == DataFormat::UInt16)                                     ? InstrModLoadStore::LO16         \
        : (DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32) ? InstrModLoadStore::INT32        \
                                                                                  : InstrModLoadStore::DEFAULT;     \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                                              \
        ckernel::sfpu::FN<APPROX_MODE, INSTRUCTION_MODE, ITERATIONS>, DST_IDX, (int)VectorMode::MODE);

// For the compare with zero ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL(OP, MODE, APPROX_MODE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(   \
        ckernel::sfpu::calculate_comp<APPROX_MODE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, 8);

// Generalized macro for compare-with-zero ops for any type (e.g., int, uint16)
#define SFPU_ZERO_KERNEL_TYPE(TYPE, OP, MODE, APPROX_MODE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(              \
        ckernel::sfpu::TYPE<APPROX_MODE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE);

// For log1p op that needs three template parameters
#define SFPU_UNARY_NO_PARAM_KERNEL_LOG1P_FN(FN, MODE, APPROX_MODE, FAST_APPROX, FP32, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(                                         \
        ckernel::sfpu::FN<APPROX_MODE, FAST_APPROX, FP32>, DST_IDX, (int)VectorMode::MODE)
