// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "tensix.h"
#include "tensix_types.h"

// #include "ckernel_cmd.h"
// #include "ckernel_enum.h"
// #include "ckernel_ops.h"

namespace ckernel
{

// Helper function to convert to underlying type
// e.g. to_underlying(MathFidelity::HiFi4) -> 4 (underlying type of MathFidelity is std::uint8_t)
template <typename T>
constexpr auto to_underlying(T t) noexcept
{
    return static_cast<std::underlying_type_t<T>>(t);
}

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
    std::uint32_t num_faces;
    std::uint32_t face_r_dim;
    std::uint32_t face_c_dim;
    bool narrow_tile;
};

// TODO: AM; rename enum values, issue #1275
enum ThreadId
{
    BriscThreadId  = 0,
    UnpackThreadId = 1,
    MathThreadId   = 2,
    PackThreadId   = 3
};

// For instructions that address lower/upper 16 bits of a register
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)

constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH  = 16;
constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH  = 32;

constexpr std::uint32_t FACE_R_DIM = FACE_HEIGHT;
constexpr std::uint32_t FACE_C_DIM = FACE_WIDTH;

constexpr std::uint32_t TILE_R_DIM = TILE_HEIGHT;
constexpr std::uint32_t TILE_C_DIM = TILE_WIDTH;

constexpr std::uint32_t TILE_NUM_FACES = ((TILE_R_DIM * TILE_C_DIM) / (FACE_R_DIM * FACE_C_DIM));

constexpr std::uint32_t DEST_NUM_TILES_FP16      = (DEST_REGISTER_FULL_SIZE * DEST_FACE_WIDTH) / (TILE_HEIGHT * TILE_HEIGHT);
constexpr std::uint32_t DEST_NUM_TILES_FP16_HALF = DEST_NUM_TILES_FP16 / 2;
static_assert((DEST_NUM_TILES_FP16 & (DEST_NUM_TILES_FP16 - 1)) == 0);

} // namespace ckernel
