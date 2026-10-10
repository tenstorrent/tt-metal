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
    // Kept as raw TTI intrinsics on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi
    // `v & 0x7FFFFFFF` form (#48598) needed two extra per-element SFPLOADI to rebuild the mask inside
    // the replay block, costing +30% cyc/tile.
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, InstrModLoadStore::INT32, 3, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, InstrModLoadStore::INT32, 3, 0);
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
