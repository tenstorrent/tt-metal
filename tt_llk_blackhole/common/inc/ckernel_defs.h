// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_ops.h"
#include "llk_defs.h"
#include "tensix.h"
#include "tensix_types.h"

namespace ckernel
{

enum Srcs
{
    SrcA = 0,
    SrcB = 1,
    SrcC = 2
};

enum Unpackers
{
    Unp0   = 0,
    Unp1   = 1,
    UnpAll = 2
};

enum DstStart
{
    StartZero = 0,
    StartHalf = 1
};

enum DstClear
{
    ClearRow    = 0,
    Clear16Rows = 1,
    ClearHalf   = 2,
    ClearFull   = 3
};

enum ThreadId
{
    BriscThreadId  = 0,
    UnpackThreadId = 1,
    MathThreadId   = 2,
    PackThreadId   = 3
};

enum class DataLayout
{
    TILE      = 0,
    ROW_MAJOR = 1
};

enum DstTileShape
{
    Tile32x32 = 0,
    Tile32x16 = 1,
    Tile16x16 = 2
};

enum UnpackDestination
{
    SrcRegs = 0,
    DestReg = 1
};

enum register_space_e
{
    TDMA_REGS     = 0x0,
    LOCAL_REGS    = 0x1,
    ADDR_COUNTERS = 0x2
};

enum SortDir : bool
{
    ArgMax = false,
    ArgMin = true,
};

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

// For instructions that address lower/upper 16 bits of a register
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)

// Helper function to convert to underlying type
// e.g. to_underlying(MathFidelity::HiFi4) -> 4 (underlying type of MathFidelity is std::uint8_t)
template <typename T>
constexpr auto to_underlying(T t) noexcept
{
    return static_cast<std::underlying_type_t<T>>(t);
}

constexpr static std::uint32_t GET_L1_HEADERLESS_TILE_SIZE(std::uint32_t format)
{
    switch (format & 0xF)
    {
        case (to_underlying(DataFormat::Int32)):
        case (to_underlying(DataFormat::Float32)):
            return (4096 >> 4);
        case (to_underlying(DataFormat::Float16)):
        case (to_underlying(DataFormat::Float16_b)):
            return (2048 >> 4);
        case (to_underlying(DataFormat::Bfp8)):
        case (to_underlying(DataFormat::Bfp8_b)):
            return ((1024 >> 4) + (64 >> 4));
        case (to_underlying(DataFormat::Bfp4)):
        case (to_underlying(DataFormat::Bfp4_b)):
            return ((512 >> 4) + (64 >> 4));
        case (to_underlying(DataFormat::Bfp2)):
        case (to_underlying(DataFormat::Bfp2_b)):
            return ((256 >> 4) + (64 >> 4));
        case (to_underlying(DataFormat::Int8)):
        case (to_underlying(DataFormat::Lf8)):
        case (to_underlying(DataFormat::Fp8_e4m3)):
            return (1024 >> 4);
        default:
            return ((1024 >> 4) + (64 >> 4));
    };
}

constexpr static bool IS_BFP_FORMAT(std::uint32_t format)
{
    switch (format & 0xF)
    {
        case (to_underlying(DataFormat::Bfp8)):
        case (to_underlying(DataFormat::Bfp8_b)):
        case (to_underlying(DataFormat::Bfp4)):
        case (to_underlying(DataFormat::Bfp4_b)):
        case (to_underlying(DataFormat::Bfp2)):
        case (to_underlying(DataFormat::Bfp2_b)):
            return true;
        default:
            return false;
    };
}

constexpr static bool IS_BFP_A_FORMAT(std::uint32_t format)
{
    switch (format & 0xF)
    {
        case (to_underlying(DataFormat::Bfp8)):
        case (to_underlying(DataFormat::Bfp4)):
        case (to_underlying(DataFormat::Bfp2)):
            return true;
        default:
            return false;
    };
}

constexpr static bool IS_A_FORMAT(std::uint32_t format)
{
    switch (format & 0xF)
    {
        case (to_underlying(DataFormat::Lf8)):
        case (to_underlying(DataFormat::Float16)):
        case (to_underlying(DataFormat::Bfp8)):
        case (to_underlying(DataFormat::Bfp4)):
        case (to_underlying(DataFormat::Bfp2)):
            return true;
        default:
            return false;
    };
}

constexpr static std::uint32_t SCALE_DATUM_SIZE(std::uint32_t format, std::uint32_t datum_count)
{
    switch (static_cast<DataFormat>(format & 0xF))
    {
        case DataFormat::Int32:
        case DataFormat::Float32:
            return datum_count << 2;

        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return datum_count << 1;

        default:
            return datum_count;
    };
}

#define LOWER_HALFWORD(x) ((x) & 0xFFFF)
#define UPPER_HALFWORD(x) ((x) >> 16)

enum class ActivationType
{
    Celu        = 0,
    Elu         = 1,
    Gelu        = 2,
    Hardtanh    = 3,
    Hardsigmoid = 4,
};

enum class RoundingMode : std::uint8_t
{
    None  = 0,
    Trunc = 1,
    Floor = 2,
};

enum class BinaryOp : std::uint8_t
{
    ADD           = 0,
    SUB           = 1,
    MUL           = 2,
    DIV           = 3,
    RSUB          = 4,
    POW           = 5,
    XLOGY         = 6,
    RSHFT         = 7,
    LSHFT         = 8,
    LOGICAL_RSHFT = 9,
    ADD_TOP_ROW   = 10
};

} // namespace ckernel
