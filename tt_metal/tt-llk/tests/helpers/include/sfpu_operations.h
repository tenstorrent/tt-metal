// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_add_top_row.h"
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_init.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu/ckernel_sfpu_topk.h"

// To add a new metal SFPU operation:
// 1. Include the metal header below: #include "llk_sfpu/<operation>.h"
// 2. Add the operation enum to SfpuType in llk_sfpu_types.h
// 3. Add the if constexpr branches in call_unary_sfpu_operation_init() and call_unary_sfpu_operation() below
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_log1p.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_abs.h"
#include "sfpu/ckernel_sfpu_activations.h"
#include "sfpu/ckernel_sfpu_elu.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_exp2.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_negative.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rsqrt.h"
#include "sfpu/ckernel_sfpu_shift.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_square.h"
#include "sfpu/ckernel_sfpu_threshold.h"
#include "sfpu/ckernel_sfpu_trigonometry.h"
#include "sfpu/ckernel_sfpu_binary.h"

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::sfpu;

/**
 * Calls only the init portion of a unary SFPU operation.
 * Must be paired with a subsequent call_unary_sfpu_operation() for the calculate step.
 * Delegates to llk_math_eltwise_unary_sfpu_init: the two-arg overload runs
 * _llk_math_eltwise_unary_sfpu_init_ then the per-operation init lambda;
 * the zero-arg overload runs only _llk_math_eltwise_unary_sfpu_init_.
 *
 * @tparam OPERATION The SFPU operation type to initialize
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 */
template <
    SfpuType OPERATION,
    bool APPROX_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    bool FAST_MODE      = false,
    bool STABLE_SORT    = false,
    bool CLAMP_NEGATIVE = false>
void call_unary_sfpu_operation_init()
{
    if constexpr (OPERATION == SfpuType::acosh || OPERATION == SfpuType::asinh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_inverse_hyperbolic_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::atanh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_atanh_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_exp2_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_exponential_<APPROX_MODE, 0x3F800000 /* exp_base_scale_factor */, CLAMP_NEGATIVE>);
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_gelu_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_hardsigmoid_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_log_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::log1p)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(log1p_init<APPROX_MODE, FAST_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_reciprocal_<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_rsqrt_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_sqrt_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(tanh_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>();
    }
}

/**
 * Calls only the calculate portion of a unary SFPU operation.
 * Must be preceded by a call to call_unary_sfpu_operation_init() for the same operation.
 * Delegates to _llk_math_eltwise_unary_sfpu_params_ per operation, following the same
 * one-call-per-branch pattern as call_unary_sfpu_operation_init. Passing ITERATIONS as
 * vector_mode (≥ 8, never matching R/C/RC = 0/1/2) triggers the single-call else branch,
 * preserving existing ITERATIONS-covers-full-tile semantics.
 *
 * @tparam OPERATION The SFPU operation type to execute
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 * @param dst_index Destination tile index in the destination register
 * @param math_format Optional math format for operations that need format-specific behavior
 */
template <
    SfpuType OPERATION,
    bool APPROX_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    bool FAST_MODE      = false,
    bool STABLE_SORT    = false,
    bool CLAMP_NEGATIVE = false>
void call_unary_sfpu_operation(std::uint32_t dst_index, std::uint32_t math_format = 0, float fill_const_value = 5.0f)
{
    if constexpr (OPERATION == SfpuType::abs)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_abs_<APPROX_MODE, ITERATIONS>(ITERATIONS), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::acosh)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_acosh_<APPROX_MODE, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::asinh)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_asinh_<APPROX_MODE, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::atanh)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_atanh_<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _calculate_activation_<APPROX_MODE, ActivationType::Celu, ITERATIONS>(10 /* alpha */, 1.0f / 10.0f /* 1.0f / alpha */), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::cosine)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_cosine_<APPROX_MODE, ITERATIONS>(ITERATIONS), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_elu_<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>(1 /* alpha */), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_exp2_<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, ITERATIONS);
    }
    // VectorMode::RC: params drives 4 face iterations with 2×SETRWC between each —
    // the lambda processes 8 rows per face, giving 32 total.
    else if constexpr (OPERATION == SfpuType::exponential && APPROX_MODE && CLAMP_NEGATIVE)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _calculate_exponential_<APPROX_MODE, false /* scale_en */, 8, CLAMP_NEGATIVE>(p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */),
            dst_index,
            static_cast<int>(VectorMode::RC));
    }
    // Single call (else branch): _calculate_exponential_ handles 8 or 32 iterations internally.
    else if constexpr (OPERATION == SfpuType::exponential && APPROX_MODE)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _calculate_exponential_<APPROX_MODE, false /* scale_en */, ITERATIONS, CLAMP_NEGATIVE>(p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */),
            dst_index,
            ITERATIONS);
    }
    // Single call (else branch): non-approx mode, handles all iterations in one call.
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _calculate_exponential_<APPROX_MODE, false /* scale_en */, ITERATIONS, CLAMP_NEGATIVE>(p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */),
            dst_index,
            ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::fill)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            _llk_math_eltwise_unary_sfpu_params_(
                _calculate_fill_int_<APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS>(static_cast<std::uint32_t>(fill_const_value)),
                dst_index,
                ITERATIONS);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            _llk_math_eltwise_unary_sfpu_params_(
                _calculate_fill_int_<APPROX_MODE, ckernel::InstrModLoadStore::LO16, ITERATIONS>(static_cast<std::uint32_t>(fill_const_value)),
                dst_index,
                ITERATIONS);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            _llk_math_eltwise_unary_sfpu_params_(
                _calculate_fill_int_<APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS>(static_cast<std::uint32_t>(fill_const_value)),
                dst_index,
                ITERATIONS);
        }
        else
        {
            _llk_math_eltwise_unary_sfpu_params_(_calculate_fill_<APPROX_MODE, ITERATIONS>(fill_const_value), dst_index, ITERATIONS);
        }
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_gelu_<APPROX_MODE, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_activation_<APPROX_MODE, ckernel::ActivationType::Hardsigmoid, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_log_<APPROX_MODE, false, ITERATIONS>(ITERATIONS, 0 /* log_base */), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::log1p)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_log1p<APPROX_MODE, FAST_MODE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::negative)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            _llk_math_eltwise_unary_sfpu_params_(_calculate_negative_int_<APPROX_MODE, ITERATIONS>(), dst_index, ITERATIONS);
        }
        else
        {
            _llk_math_eltwise_unary_sfpu_params_(_calculate_negative_<APPROX_MODE, ITERATIONS>(), dst_index, ITERATIONS);
        }
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_reciprocal_<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_rsqrt_<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE>(ITERATIONS), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_silu_<APPROX_MODE, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::sine)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sine_<APPROX_MODE, ITERATIONS>(ITERATIONS), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sqrt_<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE>(ITERATIONS), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_square_<APPROX_MODE, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_tanh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::threshold)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _calculate_threshold_<APPROX_MODE, ITERATIONS>(5.0f /* threshold_value */, 10.0f /* replacement_value */), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::topk_local_sort)
    {
        _llk_math_eltwise_unary_sfpu_params_(

            _bitonic_topk_phases_steps<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
                0 /* idir */, 5 /* i_end_phase */, 0 /* i_start_phase */, 10 /* i_end_step */, 0 /* i_start_step */),
            dst_index,
            ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::topk_merge)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_topk_merge<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(5 /* m_iter */, 10 /* k */), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::topk_rebuild)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            _bitonic_topk_rebuild<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(0 /* idir */, 5 /* m_iter */, 10 /* k */, 3 /* logk */, 0 /* skip_second */),
            dst_index,
            ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::relu_max)
    {
        _llk_math_eltwise_unary_sfpu_params_(_relu_max_<sfpi::vFloat, APPROX_MODE, ITERATIONS>(5.0f /* threshold */), dst_index, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::relu_min)
    {
        _llk_math_eltwise_unary_sfpu_params_(_relu_min_<sfpi::vFloat, APPROX_MODE, ITERATIONS>(5.0f /* threshold */), dst_index, ITERATIONS);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

/**
 * Calls only the init portion of a binary SFPU operation.
 * Must be paired with a subsequent call_binary_sfpu_operation() for the calculate step.
 * Delegates to llk_math_eltwise_binary_sfpu_init, mirroring the same pattern as
 * call_unary_sfpu_operation_init: the two-arg overload runs
 * _llk_math_eltwise_binary_sfpu_init_ then the per-operation init lambda;
 * the zero-arg overload runs only _llk_math_eltwise_binary_sfpu_init_.
 */
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation_init()
{
    if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV || BINOP == BinaryOp::RSUB ||
        BINOP == BinaryOp::XLOGY || BINOP == BinaryOp::POW)
    {
        llk_math_eltwise_binary_sfpu_init<SfpuType::add1>(_sfpu_binary_init_<APPROXIMATION_MODE, BINOP>);
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        llk_math_eltwise_binary_sfpu_init<SfpuType::add1>(_init_add_top_row_);
    }
    else
    {
        llk_math_eltwise_binary_sfpu_init<SfpuType::add1>();
    }
}

/**
 * Calls only the calculate portion of a binary SFPU operation.
 * Must be preceded by a call to call_binary_sfpu_operation_init() for the same operation.
 * Delegates to _llk_math_eltwise_binary_sfpu_params_ per operation, following the same
 * one-call-per-branch pattern. Passing ITERATIONS as vector_mode (≥ 8, never matching
 * R/C/RC = 0/1/2) triggers the single-call else branch. The callable receives
 * (dst_index_in0, dst_index_in1, dst_index_out) forwarded from the params wrapper.
 */
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation(
    std::uint32_t dst_index, const std::uint32_t dst_index_in0 = 0, const std::uint32_t dst_index_in1 = 1, const std::uint32_t dst_index_out = 0)
{
    if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV || BINOP == BinaryOp::RSUB ||
        BINOP == BinaryOp::XLOGY || BINOP == BinaryOp::POW)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(in0, in1, out), dst_index_in0, dst_index_in1, dst_index_out, ITERATIONS);
    }
    else if constexpr (BINOP == BinaryOp::RSHFT)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            _calculate_binary_right_shift_<APPROXIMATION_MODE, ITERATIONS, INT32, false>(in0, in1, out),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            ITERATIONS);
    }
    else if constexpr (BINOP == BinaryOp::LSHFT)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            _calculate_binary_left_shift_<APPROXIMATION_MODE, ITERATIONS, INT32, false>(in0, in1, out),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            ITERATIONS);
    }
    else if constexpr (BINOP == BinaryOp::LOGICAL_RSHFT)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            _calculate_logical_right_shift_<APPROXIMATION_MODE, ITERATIONS, INT32, false>(in0, in1, out),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            ITERATIONS);
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        // Use actual format when compiling for ADD_TOP_ROW tests, otherwise use Float32 as safe default for static assert
        constexpr DataFormat add_top_row_format = (BINOP == BinaryOp::ADD_TOP_ROW) ? static_cast<DataFormat>(MATH_FORMAT) : DataFormat::Float32;
        _llk_math_eltwise_binary_sfpu_params_(
            _calculate_add_top_row_<add_top_row_format>(in0, in1, out), dst_index_in0, dst_index_in1, dst_index_out, ITERATIONS);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

} // namespace test_utils
