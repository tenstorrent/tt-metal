// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

enum class RoundingOp : std::uint8_t { Ceil, Floor, Trunc, Round, StochasticRound, Frac };

// ---------------------------------------------------------------------------------------------------
// Rounding<APPROX, ROUNDING_OP, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode)           -> _calculate_{ceil,floor,trunc,stochastic_round,frac}_
//                                                  (ceil_tile, floor_tile, trunc_tile, stochastic_round_tile,
//                                                  frac_tile)
//   calculate(dst_index, vector_mode, decimals) -> _calculate_round_ (round_tile; ROUNDING_OP == Round only)
//   init()                                      -> bare init        (rounding_op_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, RoundingOp ROUNDING_OP, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Rounding
    : SfpuUnaryOp<Rounding<APPROXIMATION_MODE, ROUNDING_OP, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() {
        static_assert(ROUNDING_OP != RoundingOp::Round, "Rounding<Round> takes a decimals argument");
        if constexpr (ROUNDING_OP == RoundingOp::Ceil) {
            _calculate_ceil_<APPROXIMATION_MODE, ITERATIONS>();
        } else if constexpr (ROUNDING_OP == RoundingOp::Floor) {
            _calculate_floor_<APPROXIMATION_MODE, ITERATIONS>();
        } else if constexpr (ROUNDING_OP == RoundingOp::Trunc) {
            _calculate_trunc_<APPROXIMATION_MODE, ITERATIONS>();
        } else if constexpr (ROUNDING_OP == RoundingOp::StochasticRound) {
            _calculate_stochastic_round_<APPROXIMATION_MODE, ITERATIONS>();
        } else {
            _calculate_frac_<APPROXIMATION_MODE, ITERATIONS>();
        }
    }

    static void kernel(int decimals) {
        static_assert(ROUNDING_OP == RoundingOp::Round, "only Rounding<Round> takes a decimals argument");
        _calculate_round_<APPROXIMATION_MODE, ITERATIONS>(decimals);
    }
};

}  // namespace sfpu
}  // namespace ckernel
