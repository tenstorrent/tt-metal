// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/eltwise.h"
#include "tensor_shape.h"

namespace llk::sfpu {

/// Value @ref Mask writes wherever the mask is zero.
enum class MaskFill {
    Zero,         ///< 0, for Float16_b, Float16 and Int32
    FloatPosInf,  ///< float(inf), for Float16_b and Float16 only
};

class Mask {
private:
    static constexpr bool is_supported_integral(::DataFormat format) { return format == ::DataFormat::Int32; }

    static constexpr bool is_supported_floating(::DataFormat format) {
        return format == ::DataFormat::Float16_b || format == ::DataFormat::Float16;
    }

    template <::DataFormat format>
    using vector_type = std::conditional_t<is_supported_integral(format), sfpi::vInt, sfpi::vFloat>;

    /// @p fill as a scalar: an int 0 converts to either vector type, +inf is float only.
    template <MaskFill fill>
    static constexpr auto scalar() {
        if constexpr (fill == MaskFill::Zero) {
            return 0;
        } else if constexpr (fill == MaskFill::FloatPosInf) {
            return std::numeric_limits<float>::infinity();
        } else {
            static_assert(always_false<fill>, "Mask: unhandled MaskFill");
        }
    }

    /// False for every @p fill, so a static_assert on it fires only in the branch that is instantiated.
    template <MaskFill fill>
    static constexpr bool always_false = false;

public:
    static void init() {
        // todo(sstanisic): implement Mask::init
    }

    /**
     * @brief output = mask == 0 ? fill : input, over shape. Out of place; pass output == input to mask in place.
     *
     * @tparam shape: Region of the output, input and mask tiles to process
     * @tparam fill: Value written where the mask is zero
     * @tparam format: Float16_b and Float16 compare the mask as a float, Int32 as an integer
     */
    template <const ckernel::TensorShape& shape, MaskFill fill, ::DataFormat format>
    static void execute(std::uint32_t output, std::uint32_t input, std::uint32_t mask) {
        static_assert(
            is_supported_integral(format) || is_supported_floating(format), "SFPU Mask: Unsupported DataFormat");
        static_assert(
            fill != MaskFill::FloatPosInf || format != ::DataFormat::Int32,
            "Mask: FloatPosInf needs Float16_b or Float16");

        using vType = vector_type<format>;
        const auto value = scalar<fill>();

        eltwise_binary<shape, PaddingAgnostic::Yes>(output, input, mask, [value](vType data_vec, vType mask_vec) {
            v_if(mask_vec == 0) { data_vec = value; }
            v_endif;
            return data_vec;
        });
    }
};

}  // namespace llk::sfpu
