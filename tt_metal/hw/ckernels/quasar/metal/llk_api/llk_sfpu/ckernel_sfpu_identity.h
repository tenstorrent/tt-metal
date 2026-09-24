// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity_uint() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

// Op class for identity(x) on float tiles. Same name and leading template parameters as on
// Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Identity : SfpuUnaryOp<Identity<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = calculate_identity<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class for identity(x) on uint32 tiles. Same name and leading template parameters as on
// Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct IdentityUint : SfpuUnaryOp<IdentityUint<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = calculate_identity_uint<APPROXIMATION_MODE, ITERATIONS>;
};

}  // namespace sfpu
}  // namespace ckernel
