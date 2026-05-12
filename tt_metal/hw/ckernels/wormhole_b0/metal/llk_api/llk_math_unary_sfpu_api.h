// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_abs.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "ckernel_sfpu_exp2.h"
#include "ckernel_sfpu_expm1.h"
#include "ckernel_sfpu_heaviside.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_unary_power.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_sign.h"
#include "ckernel_sfpu_signbit.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_square.h"
#include "ckernel_sfpu_tanh.h"
#include "ckernel_sfpu_tiled_prod.h"
#include "ckernel_sfpu_topk.h"
#include "ckernel_sfpu_alt_complex_rotate90.h"
#include "ckernel_sfpu_reduce.h"

namespace ckernel {

// abs
inline void llk_math_eltwise_unary_sfpu_abs_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::abs>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_abs, (APPROXIMATE), dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_int32(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_abs_int32, (APPROXIMATE), dst_index, vector_mode);
}

// max_min
// Unary maximum
inline void llk_math_eltwise_unary_sfpu_unary_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max>(sfpu::unary_max_min_init<true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min, (true, APPROXIMATE), dst_index, vector_mode, param0);
}

inline void llk_math_eltwise_unary_sfpu_unary_max_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max_int32>(sfpu::unary_max_min_int32_init<true, false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (true, false, APPROXIMATE),
        dst_index,
        vector_mode,
        param0);
}

inline void llk_math_eltwise_unary_sfpu_unary_max_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_max_uint32>(sfpu::unary_max_min_int32_init<true, true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_max_uint32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (true, true, APPROXIMATE),
        dst_index,
        vector_mode,
        param0);
}

// Unary minimum
inline void llk_math_eltwise_unary_sfpu_unary_min_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min>(sfpu::unary_max_min_init<false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min, (false, APPROXIMATE), dst_index, vector_mode, param0);
}

inline void llk_math_eltwise_unary_sfpu_unary_min_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min_int32>(sfpu::unary_max_min_int32_init<false, false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_int32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (false, false, APPROXIMATE),
        dst_index,
        vector_mode,
        param0);
}

inline void llk_math_eltwise_unary_sfpu_unary_min_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_min_uint32>(sfpu::unary_max_min_int32_init<false, true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_unary_min_uint32(
    uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_max_min_int32,
        (false, true, APPROXIMATE),
        dst_index,
        vector_mode,
        param0);
}

// exp2
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exp2_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exp2>(sfpu::exp2_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_exp2(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_exp2, (APPROXIMATE, is_fp32_dest_acc_en), dst_index, vector_mode);
}

// expm1
template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_expm1_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::expm1>(sfpu::expm1_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_expm1(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_expm1, (APPROXIMATE, is_fp32_dest_acc_en, 8), dst_index, vector_mode);
}

// heaviside
inline void llk_math_eltwise_unary_sfpu_heaviside_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::heaviside>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_heaviside(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_heaviside, (APPROXIMATE), dst_index, vector_mode, param0);
}

// log
template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log>(sfpu::log_init<APPROXIMATE, FAST_APPROX, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_log,
        (APPROXIMATE, FAST_APPROX, false, is_fp32_dest_acc_en),
        dst_index,
        vector_mode,
        0);
}

template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_with_base_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log_with_base>(
        sfpu::log_init<APPROXIMATE, FAST_APPROX, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_log_with_base(
    uint dst_index, uint base_scale, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_log,
        (APPROXIMATE, FAST_APPROX, true, is_fp32_dest_acc_en),
        dst_index,
        vector_mode,
        base_scale);
}

// power
inline void llk_math_eltwise_unary_sfpu_power_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::power>(ckernel::sfpu::sfpu_unary_pow_init);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_power(
    uint dst_index, uint32_t exponent = 0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_power,
        (APPROXIMATE, is_fp32_dest_acc_en, 8),
        dst_index,
        vector_mode,
        exponent);
}

inline void llk_math_eltwise_unary_sfpu_power_iterative_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::power>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_power_iterative(
    uint dst_index, uint32_t exponent = 0, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_unary_power_iterative,
        (APPROXIMATE, 8),
        dst_index,
        vector_mode,
        exponent);
}

// sigmoid
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid>(sfpu::sigmoid_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_sigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sigmoid,
        (APPROXIMATE, is_fp32_dest_acc_en, 8),
        dst_index,
        vector_mode);
}

// sign
inline void llk_math_eltwise_unary_sfpu_sign_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::sign>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sign(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint exponent_size_8 = 1) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sign, (APPROXIMATE), dst_index, vector_mode, exponent_size_8);
}

// signbit
inline void llk_math_eltwise_unary_sfpu_signbit_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::signbit>(ckernel::sfpu::signbit_init);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_signbit(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_signbit, (APPROXIMATE, ITERATIONS), dst_index, vector_mode);
}

inline void llk_math_eltwise_unary_sfpu_signbit_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::signbit>(ckernel::sfpu::signbit_int32_init);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_signbit_int32(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_signbit_int32, (APPROXIMATE, ITERATIONS), dst_index, vector_mode);
}

// silu
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_silu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu>(sfpu::silu_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (is_fp32_dest_acc_en, 8), dst_index, vector_mode);
}

// square
inline void llk_math_eltwise_unary_sfpu_square_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::square>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_square(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_square, (APPROXIMATE), dst_index, vector_mode);
}

// tanh
template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh>(sfpu::tanh_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (APPROXIMATE, is_fp32_dest_acc_en, 8), dst_index, vector_mode);
}

// tiled_prod
inline void llk_math_eltwise_unary_sfpu_tiled_prod_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::tiled_prod>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tiled_prod(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tiled_prod, (APPROXIMATE), dst_index, vector_mode);
}

// topk
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::topk_local_sort>(sfpu::topk_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_local_sort(
    uint dst_index,
    int idir,
    int i_end_phase,
    int i_start_phase,
    int i_end_step,
    int i_start_step,
    int vector_mode = (int)VectorMode::RC_custom) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_bitonic_topk_phases_steps,
        (APPROXIMATE, is_fp32_dest_acc_en, STABLE_SORT),
        dst_index,
        vector_mode,
        idir,
        i_end_phase,
        i_start_phase,
        i_end_step,
        i_start_step);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool idir = false, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_merge(
    uint dst_index, int m_iter, int k, int vector_mode = (int)VectorMode::RC_custom) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_bitonic_topk_merge,
        (APPROXIMATE, is_fp32_dest_acc_en, idir, STABLE_SORT),
        dst_index,
        vector_mode,
        m_iter,
        k);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_rebuild(
    uint dst_index,
    bool idir,
    int m_iter,
    int k,
    int logk,
    int skip_second,
    int vector_mode = (int)VectorMode::RC_custom) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_bitonic_topk_rebuild,
        (APPROXIMATE, is_fp32_dest_acc_en, STABLE_SORT),
        dst_index,
        vector_mode,
        idir,
        m_iter,
        k,
        logk,
        skip_second);
}

// alt_complex_rotate90
inline void llk_math_eltwise_unary_sfpu_alt_complex_rotate90_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::alt_complex_rotate90>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_alt_complex_rotate90(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_alt_complex_rotate90, (APPROXIMATE), dst_index, vector_mode);
}

// reduce
template <PoolType pool_type, DataFormat format>
inline void llk_math_eltwise_unary_sfpu_reduce_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reduce>(sfpu::init_reduce<pool_type, format>);
}

template <PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void llk_math_eltwise_unary_sfpu_reduce(
    uint32_t dst_index, uint32_t ct_dim, uint32_t rt_dim, VectorMode vector_mode) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_reduce,
        (pool_type, reduce_dim, format),
        dst_index,
        vector_mode,
        ct_dim,
        rt_dim);
}

}  // namespace ckernel
