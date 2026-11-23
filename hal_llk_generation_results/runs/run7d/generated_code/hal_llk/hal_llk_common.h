// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the HAL-based LLK implementations in this directory.
// Everything here is plain C++ arithmetic on data-format codes; all hardware access goes through hal::*.

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_defs.h"
#include "tensix_types.h"

namespace hal_llk
{

using ckernel::FACE_C_DIM;
using ckernel::FACE_R_DIM;

// Number of FPU rows (one SrcA/SrcB block) consumed by one elementwise instruction.
constexpr std::uint32_t FPU_BLOCK_ROWS = 8;
// Rows occupied by one face in Dest and in the Src banks.
constexpr std::uint32_t FACE_ROWS = 16;
// Dest rows occupied by one tile slot (every tile shape takes a full 32x32 slot).
constexpr std::uint32_t DEST_ROWS_PER_TILE = 64;

constexpr std::uint32_t format_code(const std::uint32_t format)
{
    return format & 0xF;
}

constexpr bool is_format(const std::uint32_t format, const DataFormat expected)
{
    return (format & 0x1F) == static_cast<std::uint32_t>(expected);
}

// Formats that make the FPU run in integer mode.
constexpr bool is_int8_math_format(const std::uint32_t format)
{
    return is_format(format, DataFormat::Int8) || is_format(format, DataFormat::Int32) || is_format(format, DataFormat::UInt8);
}

// Size in bytes of one datum held in a register file for the given register format.
constexpr std::uint32_t register_datum_bytes(const std::uint32_t format)
{
    switch (format & 0x3)
    {
        case 0: // Float32 / Tf32 / Int32 families
            return 4;
        case 1: // Float16 family
            return 2;
        default:
            return 1;
    }
}

} // namespace hal_llk
