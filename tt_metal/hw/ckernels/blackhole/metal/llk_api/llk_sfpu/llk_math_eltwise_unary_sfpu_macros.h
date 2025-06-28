// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

#define SFPU_UNARY_KERNEL(OP)                                                                             \
    namespace ckernel {                                                                                   \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                                    \
    }                                                                                                     \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode);                          \
    }                                                                                                     \
    }

#define SFPU_UNARY_PARAMS_KERNEL(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                         \
    namespace ckernel {                                                                            \
    template <bool APPROXIMATE>                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                        \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(sfpu::OP##_init<APPROXIMATE>); \
    }                                                                                              \
    template <bool APPROXIMATE>                                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                  \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {                 \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                           \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS);   \
    }                                                                                              \
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

// For unary ops that also need an _int32 variant
#define SFPU_UNARY_INT32_KERNEL(OP)                                                              \
    namespace ckernel {                                                                          \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP##_int32(                                        \
        uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {                    \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::calculate_##OP##_int32<APPROXIMATE>, dst_index, vector_mode, param0); \
    }                                                                                            \
    }

// For the "compare with zero" ops (eqz, nez, ltz, gtz, lez, gez)
#define SFPU_ZERO_KERNEL(NAME, TYPE, LITERAL)                                                                       \
    namespace ckernel {                                                                                             \
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
    }                                                                                                               \
    }

// For ops that need a custom init callback but no extra arguments
#define SFPU_INIT_KERNEL(OP, INIT_CB)                                                                     \
    namespace ckernel {                                                                                   \
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
    }                                                                                                     \
    }

#define SFPU_INIT_LITERAL_KERNEL(OP, INIT_CB, LIT)                                                        \
    namespace ckernel {                                                                                   \
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
    }                                                                                                     \
    }

#define SFPU_ROUNDING_KERNEL(NAME, ITER, FP32)                                                               \
    namespace ckernel {                                                                                      \
    template <bool APPROXIMATE, int ITERATIONS = ITER, bool USE_FP32 = FP32>                                 \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index, int vector_mode = (int)VectorMode::RC) {  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                     \
            ckernel::sfpu::_calculate_##NAME##_<APPROXIMATE, ITERATIONS, USE_FP32>, dst_index, vector_mode); \
    }                                                                                                        \
    }

// For ops with exactly one extra uint parameter (and no custom init callback)
#define SFPU_UNARY_ONE_PARAM_KERNEL(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS)                    \
    namespace ckernel {                                                                          \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                      \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                           \
    }                                                                                            \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                \
        uint dst_index, EXTRA_ARG_DECL, int vector_mode = (int)VectorMode::MODE) {               \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, vector_mode, EXTRA_ARG_PASS); \
    }                                                                                            \
    }

// Trig ops with exactly one dst_index argument and RC mode
#define SFPU_TRIG_KERNEL(OP)                                                                                \
    namespace ckernel {                                                                                     \
    template <bool APPROXIMATE>                                                                             \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                                 \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                                      \
    }                                                                                                       \
    template <bool APPROXIMATE>                                                                             \
    inline void llk_math_eltwise_unary_sfpu_##OP##_op(uint dst_index) {                                     \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                    \
            ckernel::sfpu::calculate_sfpu_trig<SfpuType::OP, APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }                                                                                                       \
    }

// Inverse-hyperbolic (acosh/asinh): single init + single compute
#define SFPU_INVERSE_HYPERBOLIC_KERNEL(OP, ITER)                                                          \
    namespace ckernel {                                                                                   \
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
    }                                                                                                     \
    }

// For ops with exactly one dst_index and RC mode
#define SFPU_SIMPLE_OP_KERNEL(OP)                                                        \
    namespace ckernel {                                                                  \
    template <bool APPROXIMATE>                                                          \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                              \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();                   \
    }                                                                                    \
    template <bool APPROXIMATE>                                                          \
    inline void llk_math_eltwise_unary_sfpu_##OP##_op(uint dst_index) {                  \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                 \
            ckernel::sfpu::calculate_##OP<APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }                                                                                    \
    }
