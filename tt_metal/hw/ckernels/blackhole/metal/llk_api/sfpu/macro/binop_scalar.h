// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sanitizer/api.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/eltwise.h"
#include "sfpu/operation.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

namespace llk::sfpu {

/// Operation @ref BinopScalar applies between each element x and the scalar s.
enum class BinopScalarMode {
    Add,   ///< x + s
    Sub,   ///< x - s
    Mul,   ///< x * s
    Div,   ///< x * s: the caller passes s = 1 / divisor
    Rsub,  ///< s - x
};

class BinopScalar : public Operation {
private:
    static constexpr bool is_integral(::DataFormat format) { return format == ::DataFormat::Int32; }

    static constexpr bool is_floating(::DataFormat format) {
        return format == ::DataFormat::Float32 || format == ::DataFormat::Float16_b;
    }

    /// False for every @p mode, so a static_assert on it fires only in the branch that is instantiated.
    template <BinopScalarMode>
    static constexpr bool always_false = false;

    template <::DataFormat format>
    using vector_type = std::conditional_t<is_integral(format), sfpi::vInt, sfpi::vFloat>;

    // Int32 is two's complement in Dest; Wormhole's default int layout is sign-magnitude.
    static constexpr sfpi::DataLayout layout(::DataFormat format) {
        return is_integral(format) ? sfpi::DataLayout::I32 : sfpi::DataLayout::Default;
    }

    template <::DataFormat format>
    static vector_type<format> scalar_value(const std::uint32_t scalar) {
        if constexpr (is_integral(format)) {
            return static_cast<int>(scalar);
        } else {
            return ckernel::sfpu::Converter::as_float(scalar);
        }
    }

    // A 16-bit Dest truncates on store; Rsub rounds to bf16 to nearest even, as torch does.
    template <BinopScalarMode mode, ::DataFormat format>
    static constexpr bool rounds_to_bf16 = mode == BinopScalarMode::Rsub && format == ::DataFormat::Float16_b;

    struct NoRounding {};

    template <BinopScalarMode, ::DataFormat>
    using rounding_type = NoRounding;

    template <BinopScalarMode mode, ::DataFormat format>
    static rounding_type<mode, format> rounding() {
        return {};
    }

public:
    static void init() {
        SAN_HOOK(unsupported());
        reset_sfpu_config();
    }

    /**
     * @brief output = input <mode> scalar, over shape. Out of place; pass output == input to operate in place.
     *
     * @tparam mode: Operation between each element and the scalar
     * @tparam format: Format Dest holds: Float32, Float16_b, or Int32 (Add, Sub and Rsub only)
     * @tparam sync: Dest sync mode, values = <SyncHalf/SyncFull>
     * @tparam shape: Region of the output and input tiles to process
     * @tparam padding: Whether the call may compute on lanes outside shape. See @ref PaddingAgnostic.
     * @param output: Dest tile the result is written to
     * @param input: Dest tile read
     * @param scalar: The scalar's bits: an fp32 value, or an int32 for Int32
     */
    template <
        BinopScalarMode mode,
        ::DataFormat format,
        ckernel::DstSync sync,
        ckernel::TensorShape shape,
        PaddingAgnostic padding = PaddingAgnostic::Yes>
    [[gnu::always_inline]] static void execute(
        const std::uint32_t output, const std::uint32_t input, const std::uint32_t scalar) {
        static_assert(
            is_integral(format) || is_floating(format), "BinopScalar: Dest format must be Float32, Float16_b or Int32");
        static_assert(
            !is_integral(format) || mode == BinopScalarMode::Add || mode == BinopScalarMode::Sub ||
                mode == BinopScalarMode::Rsub,
            "BinopScalar: Int32 supports Add, Sub and Rsub only");
        assert_shape_walkable<shape>();
        assert_walk_agnostic<shape, padding>();
        LLK_VALIDATE_TENSOR_SHAPE_MATH("llk::sfpu::BinopScalar::execute", shape);
        assert_dest_index_valid<sync>(output);
        assert_dest_index_valid<sync>(input);
        SAN_HOOK(unsupported());

        using vType = vector_type<format>;
        const vType value = scalar_value<format>(scalar);
        const auto rounding = BinopScalar::rounding<mode, format>();

        eltwise_unary<shape, padding>(
            output,
            input,
            [value, rounding](const DestRegister& out, const DestRegister& in) __attribute__((always_inline)) {
                const vType x = in.mode<layout(format)>();
                vType result;
                if constexpr (mode == BinopScalarMode::Add) {
                    result = x + value;
                } else if constexpr (mode == BinopScalarMode::Sub) {
                    result = x - value;
                } else if constexpr (mode == BinopScalarMode::Rsub) {
                    result = value - x;
                } else if constexpr (mode == BinopScalarMode::Mul || mode == BinopScalarMode::Div) {
                    // Div multiplies: the caller inverts the divisor.
                    result = x * value;
                } else {
                    static_assert(always_false<mode>, "BinopScalar: unhandled BinopScalarMode");
                }
                if constexpr (rounds_to_bf16<mode, format>) {
                    result = float32_to_bf16_rne(result);
                }
                out.mode<layout(format)>() = result;
            });
    }

    static void uninit() {}
};

}  // namespace llk::sfpu
