// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"
#include "sfpu/dispatch.h"
#include "tensor_shape.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_int_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt val = sfpi::dst_reg[0];
        v_if (val != 0)
        {
            sfpi::dst_reg[0] = sfpi::as<sfpi::vInt>(-sfpi::as<sfpi::vFloat>(val));
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

/// @ref UnaryDispatch op: tile = -tile, on float data.
struct Negative : llk::sfpu::SfpuUnary {
    static constexpr llk::sfpu::PaddingAgnostic padding_agnostic() { return llk::sfpu::PaddingAgnostic::Yes; }

    struct Init : llk::sfpu::UnaryInit<Negative> {
        void run() const {}
    };

    template <const TensorShape& SHAPE, llk::sfpu::PaddingAgnostic PADDING>
    struct Execute : llk::sfpu::UnaryExecute<Negative> {
        void run() const {
            llk::sfpu::unary<PADDING>(
                llk::sfpu::DestTensor<sfpi::vFloat, SHAPE>(this->output),
                llk::sfpu::DestTensor<sfpi::vFloat, SHAPE>(this->input),
                [](const sfpi::vFloat value) { return -value; });
        }
    };
};

/// @ref UnaryDispatch op: tile = -tile, on int32 data.
struct NegativeInt : llk::sfpu::SfpuUnary {
    static constexpr llk::sfpu::PaddingAgnostic padding_agnostic() { return llk::sfpu::PaddingAgnostic::Yes; }

    struct Init : llk::sfpu::UnaryInit<NegativeInt> {
        void run() const {}
    };

    template <const TensorShape& SHAPE, llk::sfpu::PaddingAgnostic PADDING>
    struct Execute : llk::sfpu::UnaryExecute<NegativeInt> {
        void run() const {
            llk::sfpu::unary<PADDING>(
                llk::sfpu::DestTensor<sfpi::vInt, SHAPE>(this->output),
                llk::sfpu::DestTensor<sfpi::vInt, SHAPE>(this->input),
                [](const sfpi::vInt value) {
                    // Dest holds int32 as sign-magnitude: flip the sign bit, but leave 0 alone so it
                    // does not become -0.
                    sfpi::vInt result = value;
                    v_if(value != 0) { result = sfpi::as<sfpi::vInt>(-sfpi::as<sfpi::vFloat>(value)); }
                    v_endif;
                    return result;
                });
        }
    };
};

} // namespace sfpu
} // namespace ckernel
