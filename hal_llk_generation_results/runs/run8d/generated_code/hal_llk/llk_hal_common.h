// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the HAL-based LLK library in tests/hal_llk.
// Everything that touches hardware goes through the Blackhole HAL (hal::cfg, hal::sync, ...).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "tensix_types.h"

// The LLK API (and the tests calling it) name ckernel types unqualified.
using namespace ckernel;

/**
 * @brief Transform a byte L1 address to the 16-byte, off-by-one form used by L1 address fields: (address / 16) - 1.
 */
constexpr inline std::uint32_t L1_ADDRESS(const std::uint32_t buffer_address)
{
    return (buffer_address >> 4) - 1;
}
namespace hal_llk
{

// Semaphore assignment (same roles as ckernel::semaphore, expressed as HAL selectors).
inline constexpr hal::sync::Semaphore SEM_MATH_PACK   = hal::sync::Semaphore::S1; // math <-> pack sync on dest register
inline constexpr hal::sync::Semaphore SEM_UNPACK_SYNC = hal::sync::Semaphore::S5; // trisc <-> unpacker sync on config contexts

// Dest register geometry (rows of 16 datums).
inline constexpr std::uint32_t DEST_HALF_ROWS     = 512; // a dest half (SyncHalf bank) starts at row 0 or 512
inline constexpr std::uint32_t DEST_ROWS_PER_TILE = 64;  // 32x32 tile = 4 faces x 16 rows
inline constexpr std::uint32_t FACE_ROWS          = 16;
inline constexpr std::uint32_t FACE_COLS          = 16;

// Hardware data format field (the HW only has 4 bits per format).
constexpr std::uint32_t hw_format(const std::uint32_t format)
{
    return format & 0xF;
}

// Datum size in bytes as seen by register-file strides: Float32/Tf32/Int32 -> 4, Float16/Float16_b -> 2, else 1.
constexpr std::uint32_t reg_datum_bytes(const std::uint32_t format)
{
    return ((format & 0x3) == 0) ? 4 : (((format & 0x3) == 1) ? 2 : 1);
}

constexpr bool is_int8_or_int32(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1F;
    return (f == static_cast<std::uint32_t>(DataFormat::Int8)) || (f == static_cast<std::uint32_t>(DataFormat::Int32));
}

constexpr bool is_32bit_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1F;
    return (f == static_cast<std::uint32_t>(DataFormat::Float32)) || (f == static_cast<std::uint32_t>(DataFormat::Int32)) ||
           (f == static_cast<std::uint32_t>(DataFormat::UInt32));
}

constexpr bool is_bfp_a_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1F;
    return (f == static_cast<std::uint32_t>(DataFormat::Bfp8)) || (f == static_cast<std::uint32_t>(DataFormat::Bfp4)) ||
           (f == static_cast<std::uint32_t>(DataFormat::Bfp2));
}

constexpr bool is_bfp_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1F;
    return is_bfp_a_format(f) || (f == static_cast<std::uint32_t>(DataFormat::Bfp8_b)) || (f == static_cast<std::uint32_t>(DataFormat::Bfp4_b)) ||
           (f == static_cast<std::uint32_t>(DataFormat::Bfp2_b));
}

} // namespace hal_llk
