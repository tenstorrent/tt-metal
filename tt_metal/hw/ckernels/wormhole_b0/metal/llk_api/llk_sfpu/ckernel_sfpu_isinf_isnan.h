// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_isinf_isnan.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// IsInfIsNan<OP, APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode)
//   backs isinf/isposinf/isneginf/isnan/isfinite_tile and their *_tile_init entry points.
//   OP is one of IsInfNanMode::IsInf/IsPosInf/IsNegInf/IsNan/IsFinite.
// ---------------------------------------------------------------------------------------------------
template <IsInfNanMode OP, bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct IsInfIsNan
    : SfpuUnaryOp<IsInfIsNan<OP, APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static_assert(
        OP == IsInfNanMode::IsInf || OP == IsInfNanMode::IsPosInf || OP == IsInfNanMode::IsNegInf ||
            OP == IsInfNanMode::IsNan || OP == IsInfNanMode::IsFinite,
        "IsInfIsNan supports only IsInf / IsPosInf / IsNegInf / IsNan / IsFinite");

    static void kernel() { _calculate_sfpu_isinf_isnan_<OP, APPROXIMATION_MODE, ITERATIONS>(); }
};

}  // namespace sfpu
}  // namespace ckernel
