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

// For erf and erfc which share one compute callback
#define SFPU_ERF_ERFC_KERNEL(NAME, TYPE)                                                                          \
    namespace ckernel {                                                                                           \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                                     \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>();                                          \
    }                                                                                                             \
    template <bool APPROXIMATE>                                                                                   \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index, int param0 /*= 0*/) {                          \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                          \
            ckernel::sfpu::calculate_sfpu_erf_erfc<SfpuType::TYPE, APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }                                                                                                             \
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
    namespace ckernel {                                                                                    \
    template <bool APPROXIMATE>                                                                            \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(EXTRA_ARG_DECL) {                                  \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>, EXTRA_ARG_PASS); \
    }                                                                                                      \
    }

// For a simple compute‐only kernel for ops whose sfpu::calculate_<OP> takes two template parameters: <APPROXIMATE,
// LITERAL_ITERATIONS>
#define SFPU_SIMPLE_TWO_PARAM_KERNEL(OP, ITER)                                                            \
    namespace ckernel {                                                                                   \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE, ITER>, dst_index, vector_mode);                    \
    }                                                                                                     \
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
    namespace ckernel {                                                                        \
    template <bool APPROXIMATE>                                                                \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                    \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(INIT_CB<APPROXIMATE>); \
    }                                                                                          \
    }

#define SFPU_DIM_SWITCH_KERNEL(OP, ENUM, CALC0, MODE0, CALC1, MODE1)                   \
    namespace ckernel {                                                                \
    template <bool APPROXIMATE>                                                        \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, ENUM dim) {           \
        if (dim == ENUM::SUM_COL) {                                                    \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                           \
                ckernel::sfpu::CALC0<APPROXIMATE>, dst_index, (int)VectorMode::MODE0); \
        } else {                                                                       \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                           \
                ckernel::sfpu::CALC1<APPROXIMATE>, dst_index, (int)VectorMode::MODE1); \
        }                                                                              \
    }                                                                                  \
    }

#define SFPU_ONE_PARAM_CONST_ITERS_KERNEL(OP, ITERS)                                             \
    namespace ckernel {                                                                          \
    template <bool APPROXIMATE>                                                                  \
    inline void llk_math_eltwise_unary_sfpu_##OP(                                                \
        uint dst_index, uint extra, int /*iterations*/, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                         \
            ckernel::sfpu::OP<APPROXIMATE, ITERS>, dst_index, vector_mode, extra);               \
    }                                                                                            \
    }

// For isinf / isposinf / isneginf / isnan / isfinite - they all share the same calculate_sfpu_isinf_isnan<...> callback
#define SFPU_ISINF_ISNAN_KERNEL(NAME, TYPE)                                                                          \
    namespace ckernel {                                                                                              \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##NAME##_init() {                                                        \
        llk_math_eltwise_unary_sfpu_init<SfpuType::TYPE, APPROXIMATE>();                                             \
    }                                                                                                                \
    template <bool APPROXIMATE>                                                                                      \
    inline void llk_math_eltwise_unary_sfpu_##NAME(uint dst_index) {                                                 \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                             \
            ckernel::sfpu::calculate_sfpu_isinf_isnan<SfpuType::TYPE, APPROXIMATE>, dst_index, (int)VectorMode::RC); \
    }                                                                                                                \
    }

// For logical_not_unary (float + int32 variants) which share one init and two custom callbacks
#define SFPU_LOGICAL_NOT_NOTI_KERNEL(NAME, VT0, ET0, VT1, ET1)                          \
    namespace ckernel {                                                                 \
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
    }                                                                                   \
    }

// For the simple "init" + "<OP>_op" pair
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

// For a pair of compute‐only ops that switch on an enum
#define SFPU_DIM_SWITCH_KERNEL(OP, ENUM, CALC0, MODE0, CALC1, MODE1)                   \
    namespace ckernel {                                                                \
    template <bool APPROXIMATE>                                                        \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, ENUM dim) {           \
        if (dim == ENUM::MODE0) {                                                      \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                           \
                ckernel::sfpu::CALC0<APPROXIMATE>, dst_index, (int)VectorMode::MODE0); \
        } else {                                                                       \
            llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                           \
                ckernel::sfpu::CALC1<APPROXIMATE>, dst_index, (int)VectorMode::MODE1); \
        }                                                                              \
    }                                                                                  \
    }

// For a rand kernel needs both a custom init(seed) and a fixed-RC compute(from,scale)
#define SFPU_RAND_KERNEL(OP, INIT_CB)                                                                 \
    namespace ckernel {                                                                               \
    template <bool APPROXIMATE>                                                                       \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(uint32_t seed = 0) {                          \
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(INIT_CB<APPROXIMATE>, seed);  \
    }                                                                                                 \
    template <bool APPROXIMATE>                                                                       \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint32_t dst_index, uint32_t from, uint32_t scale) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                              \
            ckernel::sfpu::OP<APPROXIMATE>, dst_index, (int)VectorMode::RC, from, scale);             \
    }                                                                                                 \
    }

// For ops with two template parameters <APPROXIMATE, is_fp32_dest_acc_en> and a fixed third template arg = 8
#define SFPU_TEMPLATE_TWO_PARAM_KERNEL(OP, INIT_CB)                                                       \
    namespace ckernel {                                                                                   \
    template <bool APPROXIMATE>                                                                           \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                                               \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>);                \
    }                                                                                                     \
    template <bool APPROXIMATE, bool is_fp32_dest_acc_en>                                                 \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index, int vector_mode = (int)VectorMode::RC) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                  \
            ckernel::sfpu::calculate_##OP<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode);  \
    }                                                                                                     \
    }

// For all of the ReLU variants (leaky, min, max) which take a single uint param0
// plus a pure relu == relu_min(dst,0).
#define SFPU_RELU_VARIANTS()                                                        \
    /* lrelu      */                                                                \
    SFPU_UNARY_ONE_PARAM_KERNEL(lrelu, RC, uint param0 = 0, param0)                 \
    /* relu_max   */                                                                \
    SFPU_UNARY_ONE_PARAM_KERNEL(relu_max, RC, uint param0 = 0, param0)              \
    /* relu_min   */                                                                \
    SFPU_UNARY_ONE_PARAM_KERNEL(relu_min, RC, uint param0 = 0, param0)              \
    /* relu init  */                                                                \
    namespace ckernel {                                                             \
    template <bool APPROXIMATE>                                                     \
    inline void llk_math_eltwise_unary_sfpu_relu_init() {                           \
        llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROXIMATE>();        \
    }                                                                               \
    /* pure relu = relu_min(dst,0) */                                               \
    template <bool APPROXIMATE>                                                     \
    inline void llk_math_eltwise_unary_sfpu_relu(uint dst_index) {                  \
        llk_math_eltwise_unary_sfpu_relu_min<APPROXIMATE>(dst_index, /*param0=*/0); \
    }                                                                               \
    }

// For a kernel with exactly two extra uint parameters (init and compute)
#define SFPU_UNARY_TWO_PARAM_KERNEL(OP, MODE, INIT_CB, EXTRA1_DECL, EXTRA2_DECL, EXTRA1_PASS, EXTRA2_PASS)           \
    namespace ckernel {                                                                                              \
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
    }                                                                                                                \
    }

#define SFPU_RSUB_KERNEL()                                                                             \
    namespace ckernel {                                                                                \
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
    }                                                                                                  \
    }
