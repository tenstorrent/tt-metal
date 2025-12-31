// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_add_top_row.h"
#include "ckernel_sfpu_binary.h"
#include "llk_sfpu_types.h"

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::sfpu;

/**
 * Template function to call SFPU operations with parameterized iteration count
 * and optional math format for type-specific behavior.
 *
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 * @param operation The SFPU operation type to execute
 * @param math_format Optional math format for operations that need format-specific behavior
 */
template <bool APPROX_MODE, bool is_fp32_dest_acc_en, int ITERATIONS, bool FAST_MODE = false, bool STABLE_SORT = false>
void call_sfpu_operation(SfpuType operation, uint32_t math_format = 0)
{
    switch (operation)
    {
        case SfpuType::abs:
            _calculate_abs_<APPROX_MODE, ITERATIONS>(ITERATIONS);
            break;
        case SfpuType::acosh:
            _init_inverse_hyperbolic_<APPROX_MODE>();
            _calculate_acosh_<APPROX_MODE, ITERATIONS>();
            break;
        case SfpuType::asinh:
            _init_inverse_hyperbolic_<APPROX_MODE>();
            _calculate_asinh_<APPROX_MODE, ITERATIONS>();
            break;
        case SfpuType::atanh:
            _init_atanh_<APPROX_MODE>();
            _calculate_atanh_<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>();
            break;
        case SfpuType::celu:
            _calculate_activation_<APPROX_MODE, ActivationType::Celu, ITERATIONS>(10, 1.0f / 10.0f);
            break;
        case SfpuType::cosine:
            _calculate_cosine_<APPROX_MODE, ITERATIONS>(ITERATIONS);
            break;
        case SfpuType::elu:
            _init_elu_<APPROX_MODE>();
            _calculate_elu_<APPROX_MODE, ITERATIONS>(1);
            break;
        case SfpuType::exp2:
            _init_exp2_<APPROX_MODE>();
            _calculate_exp2_<APPROX_MODE, ITERATIONS>();
            break;
        case SfpuType::exponential:
            _init_exponential_<APPROX_MODE, FAST_MODE, 0x3F800000 /* exp_base_scale_factor */>();
            _calculate_exponential_<APPROX_MODE, false /* scale_en */, ITERATIONS, FAST_MODE, false /* skip_positive_check */>(
                p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
            break;
        case SfpuType::fill:
            if (math_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32))
            {
                _calculate_fill_int_<APPROX_MODE, ITERATIONS>(5);
            }
            else
            {
                _calculate_fill_<APPROX_MODE, ITERATIONS>(5.0f);
            }
            break;
        case SfpuType::gelu:
            _init_gelu_<APPROX_MODE>();
            _calculate_gelu_<APPROX_MODE, ITERATIONS>();
            break;
        case SfpuType::hardsigmoid:
            _init_hardsigmoid_<APPROX_MODE>();
            _calculate_activation_<APPROX_MODE, ckernel::ActivationType::Hardsigmoid, ITERATIONS>();
            break;
        case SfpuType::log:
            _init_log_<APPROX_MODE>();
            _calculate_log_<APPROX_MODE, false, ITERATIONS>(ITERATIONS, 0);
            break;
        case SfpuType::neg:
        case SfpuType::negative:
            if (math_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32))
            {
                _calculate_negative_int_<APPROX_MODE, ITERATIONS>();
            }
            else
            {
                _calculate_negative_<APPROX_MODE, ITERATIONS>();
            }
            break;
        case SfpuType::reciprocal:
            _init_reciprocal_<APPROX_MODE, is_fp32_dest_acc_en>();
            _calculate_reciprocal_<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
            break;
        case SfpuType::rsqrt:
            _init_rsqrt_<APPROX_MODE>();
            _calculate_rsqrt_<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE>(ITERATIONS);
            break;
        case SfpuType::silu:
            _calculate_silu_<APPROX_MODE, ITERATIONS>();
            break;
        case SfpuType::sine:
            _calculate_sine_<APPROX_MODE, ITERATIONS>(ITERATIONS);
            break;
        case SfpuType::sqrt:
            _init_sqrt_<APPROX_MODE>();
            _calculate_sqrt_<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE>(ITERATIONS);
            break;
        case SfpuType::square:
            _calculate_square_<APPROX_MODE, ITERATIONS>();
            break;
        case SfpuType::threshold:
            _calculate_threshold_<APPROX_MODE, ITERATIONS>(5.0f, 10.0f);
            break;
        case SfpuType::topk_local_sort:
            _bitonic_topk_phases_steps<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
                /* idir */ 0,
                /* i_end_phase */ 5,
                /* i_start_phase */ 0,
                /* i_end_step */ 10,
                /* i_start_step */ 0);
            break;
        case SfpuType::topk_merge:
            _bitonic_topk_merge<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
                /* m_iter */ 5,
                /* k */ 10);
            break;
        case SfpuType::topk_rebuild:
            _bitonic_topk_rebuild<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
                /* idir */ 0,
                /* m_iter */ 5,
                /* k */ 10,
                /* logk */ 3,
                /* skip_second */ 0);
            break;
        case SfpuType::relu_max:
            _relu_max_<sfpi::vFloat, APPROX_MODE, ITERATIONS>(5.0f);
            break;
        case SfpuType::relu_min:
            _relu_min_<sfpi::vFloat, APPROX_MODE, ITERATIONS>(5.0f);
            break;
        default:
            return; // Unsupported op – should never happen
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation(const uint dst_index_in0 = 0, const uint dst_index_in1 = 1, const uint dst_index_out = 0)
{
    switch (BINOP)
    {
        case BinaryOp::ADD:
        case BinaryOp::SUB:
        case BinaryOp::MUL:
        case BinaryOp::DIV:
        case BinaryOp::RSUB:
        case BinaryOp::XLOGY:
        case BinaryOp::POW:
            _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
            _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
            break;
        case BinaryOp::RSHFT:
            _calculate_binary_right_shift_<APPROXIMATION_MODE, ITERATIONS, INT32, false>(dst_index_in0, dst_index_in1, dst_index_out);
            break;
        case BinaryOp::LSHFT:
            _calculate_binary_left_shift_<APPROXIMATION_MODE, ITERATIONS, INT32, false>(dst_index_in0, dst_index_in1, dst_index_out);
            break;
        case BinaryOp::LOGICAL_RSHFT:
            _calculate_logical_right_shift_<APPROXIMATION_MODE, ITERATIONS, INT32, false>(dst_index_in0, dst_index_in1, dst_index_out);
            break;
        case BinaryOp::ADD_TOP_ROW:
            _init_add_top_row_();
            // Use actual format when compiling for ADD_TOP_ROW tests, otherwise use Float32 as safe default for static assert
            {
                constexpr DataFormat add_top_row_format = (BINOP == BinaryOp::ADD_TOP_ROW) ? static_cast<DataFormat>(MATH_FORMAT) : DataFormat::Float32;
                _calculate_add_top_row_<add_top_row_format>(dst_index_in0, dst_index_in1, dst_index_out);
            }
            break;
        default:
            return;
    }
}

} // namespace test_utils
