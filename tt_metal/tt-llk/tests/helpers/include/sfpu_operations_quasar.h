// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"

// To add a new Quasar unary SFPU operation:
// 1. Include its `ckernel_sfpu_<op>.h` below.
// 2. Add the `SfpuType` enumerator to the `if constexpr` chain in
//    call_unary_sfpu_operation_quasar() (and to init_unary_sfpu_operation_quasar()
//    if the op needs an init step).
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_fill.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rsqrt.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"

// Binary SFPU op headers (consumed by the binary dispatchers below). The op is
// selected via the LLK ckernel::BinaryOp enum (reused like Blackhole; the
// comparison and max/min enumerators were added to it in llk_defs.h).
//
// To add a new Quasar binary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the enumerator to ckernel::BinaryOp (llk_defs.h) if it is not there.
// 3. Add the `if constexpr` branch in call_binary_sfpu_operation_quasar()
//    (and init_binary_sfpu_operation_quasar() if it needs an init step).
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"         // calculate_sfpu_binary / sfpu_binary_init (float mul/div)
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h" // calculate_binary_max_min / _init_binary_max_min_
#include "sfpu/ckernel_sfpu_add.h"                // _add_int_ (int add)
#include "sfpu/ckernel_sfpu_binary_comp.h"        // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"          // _mul_int32_ (int mul)

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

/**
 * @brief Whether OPERATION is one of the six comparison-to-zero modes.
 *
 * The comp family needs a runtime format switch (@ref dispatch_runtime_sfpu_format_quasar)
 * to pick the integer-vs-float compare path, unlike the float-only unary ops, so the
 * dispatcher special-cases it.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_zero_comp_op(SfpuType op)
{
    return op == SfpuType::equal_zero || op == SfpuType::not_equal_zero || op == SfpuType::less_than_zero || op == SfpuType::greater_than_zero ||
           op == SfpuType::less_than_equal_zero || op == SfpuType::greater_than_equal_zero;
}

/**
 * @brief Run the per-operation init step for a Quasar unary SFPU op.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @note Pair with @ref call_unary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuType OPERATION>
void init_unary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuType::gelu)
    {
        gelu_init();
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        init_square();
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        init_zero_comp();
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        init_typecast();
    }
}

/**
 * @brief Map a runtime SFPU math format to a compile-time DataFormat and invoke `fn` with it.
 *
 * `fn` is invoked as `fn(std::integral_constant<DataFormat, FMT>{})`; recover the format inside
 * it with `decltype(arg)::value`.
 *
 * @tparam Fn Generic callable accepting a `std::integral_constant<DataFormat, FMT>`.
 * @param sfpu_format SFPU math format selecting the sfpmem mode / result encoding.
 * @param fn Callable receiving the selected format as a compile-time constant.
 */
template <typename Fn>
void dispatch_runtime_sfpu_format_quasar(DataFormat sfpu_format, Fn&& fn)
{
    switch (sfpu_format)
    {
        case DataFormat::Int32:
            fn(std::integral_constant<DataFormat, DataFormat::Int32> {});
            break;
        case DataFormat::Int16:
            fn(std::integral_constant<DataFormat, DataFormat::Int16> {});
            break;
        case DataFormat::Int8:
            fn(std::integral_constant<DataFormat, DataFormat::Int8> {});
            break;
        case DataFormat::UInt16:
            fn(std::integral_constant<DataFormat, DataFormat::UInt16> {});
            break;
        case DataFormat::UInt8:
            fn(std::integral_constant<DataFormat, DataFormat::UInt8> {});
            break;
        case DataFormat::Float16:
            fn(std::integral_constant<DataFormat, DataFormat::Float16> {});
            break;
        case DataFormat::Float16_b:
            fn(std::integral_constant<DataFormat, DataFormat::Float16_b> {});
            break;
        case DataFormat::Float32:
            fn(std::integral_constant<DataFormat, DataFormat::Float32> {});
            break;
        default:
            LLK_ASSERT(false, "Unsupported Quasar runtime SFPU format");
            break;
    }
}

/**
 * @brief The compile-time fill literal for a given SFPU format.
 *
 * @tparam FMT The fill target's compile-time `DataFormat`.
 * @note Reads the non-dependent global `FILL_CONSTANT` (baked in by the `FILL_CONSTANT`
 *       template parameter); every build including this header must define it.
 */
template <DataFormat FMT>
consteval auto fill_value_quasar()
{
    if constexpr (FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32)
    {
        return static_cast<float>(FILL_CONSTANT);
    }
    else
    {
        return static_cast<std::uint32_t>(FILL_CONSTANT);
    }
}

/**
 * @brief Apply a Quasar unary SFPU op in-place on one Dest tile.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 (32-bit) mode.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU data format;
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <SfpuType OPERATION, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32)
{
    if constexpr (OPERATION == SfpuType::abs)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_abs_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_exp_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_gelu<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::relu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_relu_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_reciprocal_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sqrt_<true /* APPROX */, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_tanh<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sigmoid_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_silu_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_rsqrt_<ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_square<ITERATIONS>, dst_index);
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        dispatch_runtime_sfpu_format_quasar(
            sfpu_format,
            [dst_index](auto fmt_c)
            {
                constexpr DataFormat FMT = decltype(fmt_c)::value;
                _llk_math_eltwise_unary_sfpu_params_(calculate_zero_comp<false, FMT, OPERATION, ITERATIONS>, dst_index);
            });
    }
    else if constexpr (OPERATION == SfpuType::fill)
    {
        dispatch_runtime_sfpu_format_quasar(
            sfpu_format,
            [dst_index](auto fmt_c)
            {
                constexpr DataFormat FMT = decltype(fmt_c)::value;
                _llk_math_eltwise_unary_sfpu_params_(calculate_fill<FMT, is_fp32_dest_acc_en, ITERATIONS>, dst_index, VectorMode::RC, fill_value_quasar<FMT>());
            });
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        // Typecast is parameterized by the (input,output) DataFormat pair, which the test
        // bakes in as the compile-time constants TYPECAST_IN_FORMAT / TYPECAST_OUT_FORMAT (set
        // by the TYPECAST_FORMATS variant). The functor picks the conversion sequence from the
        // pair at compile time.
        _llk_math_eltwise_unary_sfpu_params_(calculate_typecast<TYPECAST_IN_FORMAT, TYPECAST_OUT_FORMAT, ITERATIONS>, dst_index);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported Quasar unary SFPU operation");
    }
}

constexpr bool quasar_binary_op_is_max_min(ckernel::BinaryOp op)
{
    return op == ckernel::BinaryOp::MAX || op == ckernel::BinaryOp::MIN;
}

/**
 * @brief Run the per-operation init step for a Quasar binary SFPU op.
 *
 * @tparam OP The binary op (compile-time `ckernel::BinaryOp` constant).
 * @note Pair with @ref call_binary_sfpu_operation_quasar for the calculate step.
 */
template <ckernel::BinaryOp OP>
void init_binary_sfpu_operation_quasar()
{
    if constexpr (OP == BinaryOp::MUL)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::MUL>(); // no-op for MUL; harmless on the int path
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::DIV>();
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        _init_binary_max_min_();
    }
    // ADD / SUB / GT / LT / LE / GE are stateless — no init.
}

/**
 * @brief Apply a Quasar binary SFPU op over two Dest operands into a result tile.
 *
 * @tparam OP The binary op (compile-time `ckernel::BinaryOp` constant).
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param base_dst_index Base Dest tile index (used by max/min's unary-params call).
 * @param src0_tile,src1_tile,dst_tile Operand / result tile indices.
 * @param math_format Dest math format (Int32 vs float path for MUL and max/min).
 * @note Must be preceded by @ref init_binary_sfpu_operation_quasar for the same op.
 */
template <ckernel::BinaryOp OP, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_binary_sfpu_operation_quasar(
    [[maybe_unused]] std::uint32_t base_dst_index, int src0_tile, int src1_tile, int dst_tile, [[maybe_unused]] DataFormat math_format)
{
    // Integer ops address Dest by row offset (tile index * rows-per-tile).
    constexpr int stride = NUM_FACES * FACE_R_DIM;
    const int in0_off    = src0_tile * stride;
    const int in1_off    = src1_tile * stride;
    const int out_off    = dst_tile * stride;

    if constexpr (OP == BinaryOp::ADD)
    {
        if (math_format == DataFormat::Int32)
        {
            _llk_math_eltwise_binary_sfpu_params_(_add_int_<false, ITERATIONS, DataFormat::Int32, 0, false>, in0_off, in1_off, out_off);
        }
        else
        {
            _llk_math_eltwise_binary_sfpu_params_(
                calculate_sfpu_binary<false /*APPROX*/, BinaryOp::ADD, is_fp32_dest_acc_en, ITERATIONS>, src0_tile, src1_tile, dst_tile);
        }
    }
    else if constexpr (OP == BinaryOp::SUB)
    {
        // Int32 SUB is not ported to Quasar (sub_int_sfpu.h is WH-only); float path only.
        _llk_math_eltwise_binary_sfpu_params_(
            calculate_sfpu_binary<false /*APPROX*/, BinaryOp::SUB, is_fp32_dest_acc_en, ITERATIONS>, src0_tile, src1_tile, dst_tile);
    }
    else if constexpr (OP == BinaryOp::GT)
    {
        _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, ITERATIONS, SfpuType::gt>, in0_off, in1_off, out_off);
    }
    else if constexpr (OP == BinaryOp::LT)
    {
        _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, ITERATIONS, SfpuType::lt>, in0_off, in1_off, out_off);
    }
    else if constexpr (OP == BinaryOp::LE)
    {
        _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, ITERATIONS, SfpuType::le>, in0_off, in1_off, out_off);
    }
    else if constexpr (OP == BinaryOp::GE)
    {
        _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, ITERATIONS, SfpuType::ge>, in0_off, in1_off, out_off);
    }
    else if constexpr (OP == BinaryOp::MUL)
    {
        if (math_format == DataFormat::Int32)
        {
            _llk_math_eltwise_binary_sfpu_params_(_mul_int32_<false, ITERATIONS>, in0_off, in1_off, out_off);
        }
        else
        {
            _llk_math_eltwise_binary_sfpu_params_(
                calculate_sfpu_binary<false /*APPROX*/, BinaryOp::MUL, is_fp32_dest_acc_en, ITERATIONS>, src0_tile, src1_tile, dst_tile);
        }
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            calculate_sfpu_binary<false /*APPROX*/, BinaryOp::DIV, is_fp32_dest_acc_en, ITERATIONS>, src0_tile, src1_tile, dst_tile);
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        constexpr bool IS_MAX = (OP == BinaryOp::MAX);
        // All integer formats route through the Int32 path; float / MX use Float32.
        if (math_format == DataFormat::Int32)
        {
            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_binary_max_min<DataFormat::Int32, IS_MAX, ITERATIONS>,
                base_dst_index,
                VectorMode::RC,
                static_cast<std::uint32_t>(src0_tile),
                static_cast<std::uint32_t>(src1_tile),
                static_cast<std::uint32_t>(dst_tile));
        }
        else
        {
            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_binary_max_min<DataFormat::Float32, IS_MAX, ITERATIONS>,
                base_dst_index,
                VectorMode::RC,
                static_cast<std::uint32_t>(src0_tile),
                static_cast<std::uint32_t>(src1_tile),
                static_cast<std::uint32_t>(dst_tile));
        }
    }
    else
    {
        // Catches BinaryOp values this dispatcher does not implement;
        // a compile-time static_assert can't be used here because OP is a
        // non-type param, so sizeof(OP)==0 is non-dependent and fires for every
        // instantiation (matches the runtime guard in the unary dispatcher).
        LLK_ASSERT(false, "Unsupported Quasar binary SFPU operation");
    }
}

} // namespace test_utils
