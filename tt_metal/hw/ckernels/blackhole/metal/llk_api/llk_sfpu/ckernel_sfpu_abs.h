// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/dispatch.h"
#include "tensor_shape.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void abs_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

/// @ref UnaryDispatch op: tile = |tile|, on float data.
struct Abs : llk::sfpu::SfpuUnary {
    static constexpr llk::sfpu::PaddingAgnostic padding_agnostic() { return llk::sfpu::PaddingAgnostic::Yes; }

    struct Init : llk::sfpu::UnaryInit<Abs> {
        void run() const {}
    };

    template <const TensorShape& SHAPE, llk::sfpu::PaddingAgnostic PADDING>
    struct Execute : llk::sfpu::UnaryExecute<Abs> {
        void run() const {
            llk::sfpu::unary<PADDING>(
                llk::sfpu::DestTensor<sfpi::vFloat, SHAPE>(this->output),
                llk::sfpu::DestTensor<sfpi::vFloat, SHAPE>(this->input),
                [](const sfpi::vFloat value) { return sfpi::abs(value); });
        }
    };
};

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // sfpi::abs(vInt) lowers to the dedicated SFPABS integer instruction (mod 0),
        // matching the raw TTI sequence (SFPLOAD + SFPABS + SFPSTORE = 3 SFPU ops).
        // On Blackhole INT32_2S_COMP load/store is a no-op vs INT32, so I32 access is
        // byte-for-byte equivalent to the previous mode-12 access. abs() yields a vMag,
        // which stores through the M32 layout.
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::M32>() = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
