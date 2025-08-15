// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensix.h"

// #include "tensix_types.h"
// #include "ckernel_cmd.h"
// #include "ckernel_enum.h"
// #include "ckernel_ops.h"

namespace ckernel
{

enum register_space_e
{
    TDMA_REGS     = 0x0,
    LOCAL_REGS    = 0x1,
    ADDR_COUNTERS = 0x2
};

// This struct contains all the information needed to specify a tile
//  shape, for a default 32x32 tile these are the values:
//  num_faces = 4;
//  face_r_dim = 16;
//  face_c_dim = 16;
//  narrow_tile = 0;
struct TileShape
{
    uint32_t num_faces;
    uint32_t face_r_dim;
    uint32_t face_c_dim;
    bool narrow_tile;
};

// For instructions that address lower/upper 16 bits of a register
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)

} // namespace ckernel
