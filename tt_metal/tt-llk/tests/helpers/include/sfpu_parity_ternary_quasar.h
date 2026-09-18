// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu/ckernel_sfpu_addcdiv.h"
#include "llk_sfpu/ckernel_sfpu_addcmul.h"
#include "llk_sfpu/ckernel_sfpu_lerp.h"
#include "llk_sfpu/ckernel_sfpu_mac.h"
#include "llk_sfpu/ckernel_sfpu_snake_beta.h"

namespace ckernel::sfpu
{

// Test-only fixed-slot adapters for the existing UnarySfpu YAML node. The
// caller supplies a three-tile block, uses dst_dest_tile_index: 0, and packs
// all three tiles. Each adapter reads tiles 0/1/2 and overwrites only tile 0.
// The ordinary RC wrapper invokes these once per face (8 iterations per face
// for YAML iterations: 32). This is not a new general ternary YAML interface.
// Addcdiv/Addcmul use scalar 0.5; their independent TernarySFPUGolden reference
// must receive the same scalar bits. Format follows the tested Dest precision.

template <bool APPROX, bool DEST_ACC, int ITERATIONS>
inline void calculate_parity_addcdiv()
{
    constexpr DataFormat format = DEST_ACC ? DataFormat::Float32 : DataFormat::Float16_b;
    calculate_addcdiv<APPROX, DEST_ACC, format, ITERATIONS>(0, 1, 2, 0, 0x3f000000u);
}

template <bool APPROX, bool DEST_ACC, int ITERATIONS>
inline void calculate_parity_addcmul()
{
    constexpr DataFormat format = DEST_ACC ? DataFormat::Float32 : DataFormat::Float16_b;
    calculate_addcmul<APPROX, DEST_ACC, format, ITERATIONS>(0, 1, 2, 0, 0x3f000000u);
}

template <bool APPROX, bool DEST_ACC, int ITERATIONS>
inline void calculate_parity_lerp()
{
    constexpr DataFormat format = DEST_ACC ? DataFormat::Float32 : DataFormat::Float16_b;
    calculate_lerp<APPROX, DEST_ACC, format, ITERATIONS>(0, 1, 2, 0);
}

template <bool APPROX, bool DEST_ACC, int ITERATIONS>
inline void calculate_parity_snake_beta()
{
    constexpr DataFormat format = DEST_ACC ? DataFormat::Float32 : DataFormat::Float16_b;
    calculate_snake_beta<APPROX, DEST_ACC, format, ITERATIONS>(0, 1, 2, 0);
}

template <bool APPROX, bool DEST_ACC, int ITERATIONS>
inline void calculate_parity_mac()
{
    constexpr DataFormat format = DEST_ACC ? DataFormat::Float32 : DataFormat::Float16_b;
    calculate_mac<APPROX, DEST_ACC, format, ITERATIONS>(0, 1, 2, 0);
}

template <bool APPROX>
inline void init_parity_addcdiv()
{
    init_addcdiv<APPROX>();
}

template <bool APPROX>
inline void init_parity_snake_beta()
{
    snake_beta_init<APPROX>();
}

template <bool APPROX, bool DEST_ACC>
inline void init_parity_mac()
{
    constexpr DataFormat format = DEST_ACC ? DataFormat::Float32 : DataFormat::Float16_b;
    mac_init<APPROX, DEST_ACC, format>();
}

// Addcmul and Lerp need no operation-specific init; the dispatcher's standard
// _llk_math_eltwise_sfpu_init_ provides the common SFPU setup.

} // namespace ckernel::sfpu
