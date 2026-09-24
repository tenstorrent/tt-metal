// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_fill.h"

namespace ckernel::sfpu {

// Op class that fills every datum of the tile with a float value. Same name and leading template
// parameters as on Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Fill : SfpuUnaryOp<Fill<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = _calculate_fill_<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class that fills every datum of the tile with an integer value, stored with INSTRUCTION_MODE. Same name
// and leading template parameters as on Wormhole/Blackhole.
template <
    bool APPROXIMATION_MODE,
    InstrModLoadStore INSTRUCTION_MODE,
    int ITERATIONS = 8,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct FillInt : SfpuUnaryOp<FillInt<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = _calculate_fill_int_<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS>;
};

// Op class that fills every datum of the tile with the float whose bit pattern is value_bit_mask. Same name
// and leading template parameters as on Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct FillBitcast : SfpuUnaryOp<FillBitcast<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = _calculate_fill_bitcast_<APPROXIMATION_MODE, ITERATIONS>;
};

}  // namespace ckernel::sfpu
