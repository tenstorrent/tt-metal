// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_add_top_row.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_topk.h"

// To add a new metal SFPU operation:
// 1. Include the metal header below: #include "llk_sfpu/<operation>.h"
// 2. Add the operation enum to SfpuType in llk_sfpu_types.h
// 3. Add the if constexpr branches in call_unary_sfpu_operation_init() and call_unary_sfpu_operation() below
#include "llk_sfpu/ckernel_sfpu_log1p.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_abs.h"
#include "sfpu/ckernel_sfpu_activations.h"
#include "sfpu/ckernel_sfpu_binary.h"
#include "sfpu/ckernel_sfpu_elu.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_exp2.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_gelu.h"
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
 * Note: `OPERATION` is a template parameter rather than a literal enumerator,
 * so the SFPU_INIT* macros (which token-paste `::SfpuType::OP`) cannot be
 * used here — we call `llk_math_eltwise_unary_sfpu_init<OPERATION>(...)`
 * directly, which is exactly what those macros expand to.
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
 * Delegates to SFPU_CALL / SFPU_CALL_CAST from llk_math_eltwise_unary_sfpu_macros.h,
 * which funnel through ckernel::_sfpu_check_and_call_<DST_SYNC_MODE, DST_ACCUM_MODE>
 * (dst-bound LLK_ASSERT, then _llk_math_eltwise_unary_sfpu_params_). Passing
 * ITERATIONS as vector_mode (≥ 8, never matching R/C/RC = 0/1/2) triggers the
 * single-call else branch, preserving existing ITERATIONS-covers-full-tile semantics.
 *
 * DST_SYNC_MODE and DST_ACCUM_MODE are the first two template parameters, mirroring
 * the convention of the underlying SFPU macros and helpers (SFPU_CALL,
 * _sfpu_check_and_call_, etc.) where the dst-sync/accum pair always leads. They
 * are forwarded to _sfpu_check_and_call_ so the dst-bound LLK_ASSERT is computed
 * against the kernel's actual sync/accumulation mode.
 *
 * @tparam DST_SYNC_MODE Kernel's DstSync mode (drives the dst-bound assert)
 * @tparam DST_ACCUM_MODE Kernel's dest-accumulation flag (drives the dst-bound assert)
 * @tparam OPERATION The SFPU operation type to execute
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 * @param dst_index Destination tile index in the destination register
 * @param math_format Optional math format for operations that need format-specific behavior
 */
template <
    DstSync DST_SYNC_MODE,
    bool DST_ACCUM_MODE,
    SfpuType OPERATION,
    bool APPROX_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    bool FAST_MODE      = false,
    bool STABLE_SORT    = false,
    bool CLAMP_NEGATIVE = false>
void call_unary_sfpu_operation(
    std::uint32_t dst_index, std::uint32_t math_format = 0, float fill_const_value = 5.0f, int vector_mode = static_cast<int>(VectorMode::None))
{
    if constexpr (OPERATION == SfpuType::abs)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_abs_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::acosh)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_acosh_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::asinh)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_asinh_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::atanh)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_atanh_, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        // Two _calculate_activation_ overloads (runtime params vs none) — cast so Callable deduces.
        SFPU_CALL_CAST(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_activation_,
            (APPROX_MODE, ActivationType::Celu, ITERATIONS),
            (void (*)(std::uint32_t, std::uint32_t)),
            dst_index,
            vector_mode,
            10,
            (1.0f / 10.0f));
    }
    else if constexpr (OPERATION == SfpuType::cosine)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_cosine_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_elu_, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode, 1u /* alpha */);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_exp2_, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    // VectorMode::RC: params drives 4 face iterations with 2×SETRWC between each —
    // the lambda processes 8 rows per face, giving 32 total.
    else if constexpr (OPERATION == SfpuType::exponential && APPROX_MODE && CLAMP_NEGATIVE)
    {
        SFPU_CALL_MODE(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_exponential_,
            (APPROX_MODE, false /* scale_en */, 8, CLAMP_NEGATIVE),
            RC,
            dst_index,
            p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    // Single call (else branch): _calculate_exponential_ handles 8 or 32 iterations internally.
    else if constexpr (OPERATION == SfpuType::exponential && APPROX_MODE)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_exponential_,
            (APPROX_MODE, false /* scale_en */, ITERATIONS, CLAMP_NEGATIVE),
            dst_index,
            vector_mode,
            p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    // Single call (else branch): non-approx mode, handles all iterations in one call.
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_exponential_,
            (APPROX_MODE, false /* scale_en */, ITERATIONS, CLAMP_NEGATIVE),
            dst_index,
            vector_mode,
            p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    else if constexpr (OPERATION == SfpuType::fill)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SFPU_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _calculate_fill_int_,
                (APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS),
                dst_index,
                vector_mode,
                static_cast<std::uint32_t>(fill_const_value));
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SFPU_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _calculate_fill_int_,
                (APPROX_MODE, ckernel::InstrModLoadStore::LO16, ITERATIONS),
                dst_index,
                vector_mode,
                static_cast<std::uint32_t>(fill_const_value));
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SFPU_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _calculate_fill_int_,
                (APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS),
                dst_index,
                vector_mode,
                static_cast<std::uint32_t>(fill_const_value));
        }
        else
        {
            SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, fill_const_value);
        }
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_gelu_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        // Zero-arg _calculate_activation_ vs param overload — cast so Callable deduces.
        SFPU_CALL_CAST(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_activation_,
            (APPROX_MODE, ckernel::ActivationType::Hardsigmoid, ITERATIONS),
            (void (*)()),
            dst_index,
            vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_log_,
            (APPROX_MODE, false, ITERATIONS),
            dst_index,
            vector_mode,
            ITERATIONS,
            0u /* log_base_scale_factor */);
    }
    else if constexpr (OPERATION == SfpuType::log1p)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log1p, (APPROX_MODE, FAST_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::negative)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_negative_int_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
        }
        else
        {
            SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_negative_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_reciprocal_, (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        SFPU_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_rsqrt_, (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_silu_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sine)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_sine_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        SFPU_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_sqrt_, (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_square_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::threshold)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_threshold_,
            (APPROX_MODE, ITERATIONS, float),
            dst_index,
            vector_mode,
            5.0f /* threshold_value */,
            10.0f /* replacement_value */);
    }
    else if constexpr (OPERATION == SfpuType::topk_local_sort)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_topk_phases_steps,
            (APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT),
            dst_index,
            vector_mode,
            0 /* idir */,
            5 /* i_end_phase */,
            0 /* i_start_phase */,
            10 /* i_end_step */,
            0 /* i_start_step */);
    }
    else if constexpr (OPERATION == SfpuType::topk_merge)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_topk_merge,
            (APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT),
            dst_index,
            vector_mode,
            5 /* m_iter */,
            10 /* k */);
    }
    else if constexpr (OPERATION == SfpuType::topk_rebuild)
    {
        SFPU_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_topk_rebuild,
            (APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT),
            dst_index,
            vector_mode,
            false /* idir */,
            5 /* m_iter */,
            10 /* k */,
            3 /* logk */,
            0 /* skip_second */);
    }
    else if constexpr (OPERATION == SfpuType::relu_max)
    {
        // Fourth template param T (threshold scalar type) — float vs uint32_t overloads.
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _relu_max_, (sfpi::vFloat, APPROX_MODE, ITERATIONS, float), dst_index, vector_mode, 5.0f /* threshold */);
    }
    else if constexpr (OPERATION == SfpuType::relu_min)
    {
        SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _relu_min_, (sfpi::vFloat, APPROX_MODE, ITERATIONS, float), dst_index, vector_mode, 5.0f /* threshold */);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

/**
 * Calls only the init portion of a binary SFPU operation.
 * Must be paired with a subsequent call_binary_sfpu_operation() for the calculate step.
 * Uses SFPU_BINARY_INIT* macros from llk_math_eltwise_binary_sfpu_macros.h, which
 * wrap `ckernel::llk_math_eltwise_binary_sfpu_init<::SfpuType::OP>(...)`.
 */
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation_init()
{
    if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV || BINOP == BinaryOp::RSUB ||
        BINOP == BinaryOp::XLOGY || BINOP == BinaryOp::POW)
    {
        SFPU_BINARY_INIT_CB(add1, _sfpu_binary_init_, (APPROXIMATION_MODE, BINOP));
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        SFPU_BINARY_INIT_FN(add1, _init_add_top_row_);
    }
    else
    {
        SFPU_BINARY_INIT(add1);
    }
}

/**
 * Calls only the calculate portion of a binary SFPU operation.
 * Must be preceded by a call to call_binary_sfpu_operation_init() for the same operation.
 * Uses SFPU_BINARY_CALL from llk_math_eltwise_binary_sfpu_macros.h, which funnels
 * through ckernel::_sfpu_binary_check_and_call_<DST_SYNC_MODE, DST_ACCUM_MODE>
 * (dst-bound LLK_ASSERTs, then _llk_math_eltwise_binary_sfpu_params_). The callable
 * receives (dst_index_in0, dst_index_in1, dst_index_out) forwarded from the params
 * wrapper.
 *
 * DST_SYNC_MODE and DST_ACCUM_MODE are the first two template parameters (matching
 * the SFPU_BINARY_CALL / _sfpu_binary_check_and_call_ convention) so the dst-bound
 * LLK_ASSERTs run against the kernel's actual sync/accumulation mode.
 */
template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation(
    const std::uint32_t dst_index_in0 = 0,
    const std::uint32_t dst_index_in1 = 1,
    const std::uint32_t dst_index_out = 0,
    int vector_mode                   = static_cast<int>(VectorMode::RC))
{
    // NOTE: The functions invoked via SFPU_BINARY_CALL below run inside
    // _llk_math_eltwise_binary_sfpu_params_, which already loops over 4 faces
    // (for VectorMode::RC) and emits 2x TTI_SETRWC cr_d 8 between calls to
    // advance the dst-write counter. The per-call inner ITERATIONS must
    // therefore be 8 (one face's worth of SFPU rows), not 32 (a full tile),
    // matching how every production llk_math_eltwise_binary_sfpu_* wrapper
    // dispatches into _calculate_sfpu_binary_ / _calculate_*_shift_.
    constexpr int PER_FACE_ITERATIONS = 8;
    if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV || BINOP == BinaryOp::RSUB ||
        BINOP == BinaryOp::XLOGY || BINOP == BinaryOp::POW)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_sfpu_binary_,
            (APPROXIMATION_MODE, BINOP, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::RSHFT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_binary_right_shift_,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, INT32, false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LSHFT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_binary_left_shift_,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, INT32, false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LOGICAL_RSHFT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_logical_right_shift_,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, INT32, false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        // Use actual format when compiling for ADD_TOP_ROW tests, otherwise use Float32 as safe default for static assert
        constexpr DataFormat add_top_row_format = (BINOP == BinaryOp::ADD_TOP_ROW) ? static_cast<DataFormat>(MATH_FORMAT) : DataFormat::Float32;
        // ADD_TOP_ROW addresses all four faces of the destination tile itself
        // (via absolute tile_offset_* in TT_SFPLOAD/TT_SFPSTORE), so it must be
        // invoked exactly once. Force VectorMode::RC_custom so
        // _llk_math_eltwise_binary_sfpu_params_ takes its single-call branch
        // and does not emit the per-face TTI_SETRWC cr_d 8 pair that would
        // otherwise shift the dst write base between calls. This matches the
        // production wrapper llk_math_eltwise_binary_sfpu_add_top_row.h.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_add_top_row_,
            (add_top_row_format),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            static_cast<int>(VectorMode::RC_custom));
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

} // namespace test_utils
