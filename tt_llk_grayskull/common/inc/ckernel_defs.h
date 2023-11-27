/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include "llk_defs.h"
#include "ckernel_ops.h"
#include "tensix_types.h"

namespace ckernel
{

enum Srcs { SrcA = 0, SrcB = 1, SrcC = 2 };
enum Unpackers { Unp0 = 0, Unp1 = 1, UnpAll = 2 };
enum DstStart { StartZero = 0, StartHalf = 0};
enum DstClear { ClearRow = 0, Clear16Rows = 1, ClearHalf = 2, ClearFull = 3 };
enum ThreadId { UnpackThreadId = 0, MathThreadId = 1, PackThreadId = 2};

enum DstTileLayout
{
    Default,
    Interleaved,
    // TightDest,
    // Conv3x3,
    // Conv1x1,
    // L1ReadSource,
    // NLLLoss,
    // IndexAccumulate, //Add polling before packing to L1
};

enum DstTileFaceLayout
{
    RowMajor, // default
    ColMajor,
};

enum DstTileShape
{
    Tile32x32 = 0,
    Tile32x16 = 1,
    Tile16x16 = 2
};
enum class ParallelPackerMode
{
    Disabled,
    SingleFTEntry,
    MultiFTEntry,
    TileParallel
};

enum register_space_e
{
    TDMA_REGS = 0x0,
    LOCAL_REGS = 0x1,
    ADDR_COUNTERS = 0x2
};

enum PackSelMask
{
    PACK_ALL=0xF, // default
    PACK_0=0x1,
    PACK_1=0x2,
    PACK_2=0x4,
    PACK_3=0x8,
    PACK_01=0x3,
    PACK_23=0xC
};

#define FACE_HEIGHT (16)
#define FACE_WIDTH  (16)
#define TILE_HEIGHT (32)
#define TILE_WIDTH  (32)
#define DATUMS_PER_ROW (16)
#define TILE_HEADER_SIZE (1)

constexpr std::uint32_t FACE_R_DIM = FACE_HEIGHT;
constexpr std::uint32_t FACE_C_DIM = FACE_WIDTH ;

constexpr std::uint32_t TILE_R_DIM = TILE_HEIGHT;
constexpr std::uint32_t TILE_C_DIM = TILE_WIDTH ;

// For instructions that address lower/upper 16 bits of a register
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)

#define SCALE_DATUM_SIZE(format,datum_count) (((format&0x3) == (uint8_t)DataFormat::Float32) ? (datum_count<<2) : (datum_count<<1))

#define LOWER_HALFWORD(x) ((x) & 0xFFFF)
#define UPPER_HALFWORD(x) ((x) >> 16) 

} // namespace ckernel
