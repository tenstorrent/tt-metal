// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the HAL-based LLK implementation (Blackhole).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "tensix_types.h"

using namespace ckernel;

namespace hal_llk
{

// Semaphores shared by the three threads (same assignment as the ckernel semaphore map).
inline constexpr hal::sync::Semaphore MATH_PACK_SEMAPHORE   = hal::sync::Semaphore::S1;
inline constexpr hal::sync::Semaphore UNPACK_SYNC_SEMAPHORE = hal::sync::Semaphore::S5;

// Number of unpacker configuration contexts used for double buffering the per-tile L1 addresses.
inline constexpr std::uint32_t UNPACK_NUM_CONTEXTS = 2;

// Dest addressing (16-bit row units): a 32x32 tile spans 64 rows, the second half of Dest starts at row 512.
inline constexpr std::uint32_t DEST_TILE_ROWS_LOG2 = 6;
inline constexpr std::uint32_t DEST_HALF_BASE_ROW  = 512;

inline constexpr std::uint32_t format_code(const std::uint32_t data_format)
{
    return data_format & 0xF;
}

// Size in bytes of one register datum for the given register-side format (Float32/Tf32: 4, Float16*: 2, else 1).
inline constexpr std::uint32_t register_datum_bytes(const std::uint32_t data_format)
{
    const std::uint32_t family = data_format & 0x3;
    return family == static_cast<std::uint32_t>(DataFormat::Float32) ? 4 : (family == static_cast<std::uint32_t>(DataFormat::Float16) ? 2 : 1);
}

inline constexpr bool is_int8_math_format(const std::uint32_t data_format)
{
    const std::uint32_t f = data_format & 0x1F;
    return f == static_cast<std::uint32_t>(DataFormat::Int8) || f == static_cast<std::uint32_t>(DataFormat::UInt8) ||
           f == static_cast<std::uint32_t>(DataFormat::Int32);
}

inline constexpr bool is_32bit_register_format(const std::uint32_t data_format)
{
    const std::uint32_t f = data_format & 0x1F;
    return f == static_cast<std::uint32_t>(DataFormat::Float32) || f == static_cast<std::uint32_t>(DataFormat::Int32) ||
           f == static_cast<std::uint32_t>(DataFormat::UInt32);
}

inline constexpr bool is_bfp_format(const std::uint32_t data_format)
{
    const std::uint32_t f = data_format & 0x1F;
    return f == static_cast<std::uint32_t>(DataFormat::Bfp8) || f == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           f == static_cast<std::uint32_t>(DataFormat::Bfp4) || f == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           f == static_cast<std::uint32_t>(DataFormat::Bfp2) || f == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
}

// Number of 8-row FPU blocks needed to cover one face.
inline constexpr std::uint32_t face_row_blocks(const std::uint32_t face_r_dim)
{
    return face_r_dim > 8 ? 2 : 1;
}

} // namespace hal_llk
