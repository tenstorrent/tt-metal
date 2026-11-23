// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the HAL-based LLK implementations in tests/hal_llk.
// Only HAL entry points (hal/*.h) are used to touch hardware state.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/unpack.h"
#include "hal/utils/address_modifier.h"
#include "llk_defs.h"
#include "llk_memory_checks.h" // L1_ADDRESS(): byte address -> 16B-granule L1 address field value
#include "tensor_shape.h"

// The LLK headers expose ckernel names at global scope; the tests rely on that.
using namespace ckernel;

namespace hal_llk
{

// Semaphore used for the math <-> pack handshake on the Dest register (ckernel::semaphore::MATH_PACK).
constexpr hal::sync::Semaphore MATH_PACK_SEMAPHORE = hal::sync::Semaphore::S1;

// Dest rows per 32x32 tile (4 faces x 16 rows); tiles are always laid out at this pitch in Dest.
constexpr std::uint32_t DEST_ROWS_PER_TILE = 64;
// Dest rows in one half of the register (SyncHalf double buffering).
constexpr std::uint32_t DEST_HALF_ROWS = 512;
// Rows processed by one FPU elementwise instruction.
constexpr std::uint32_t FPU_ROWS = 8;

constexpr std::uint32_t format_code(const std::uint32_t format)
{
    return format & 0xF;
}

// Datum size in bytes from the low two format bits: Float32/Tf32/Int32 -> 4, Float16/Float16_b -> 2, else 1.
constexpr std::uint32_t datum_bytes(const std::uint32_t format)
{
    return ((format & 0x3) == static_cast<std::uint32_t>(DataFormat::Float32))   ? 4
           : ((format & 0x3) == static_cast<std::uint32_t>(DataFormat::Float16)) ? 2
                                                                                  : 1;
}

constexpr bool is_int8_or_int32(const std::uint32_t format)
{
    return format_code(format) == format_code(static_cast<std::uint32_t>(DataFormat::Int8)) ||
           format_code(format) == format_code(static_cast<std::uint32_t>(DataFormat::Int32));
}

constexpr bool is_32bit_format(const std::uint32_t format)
{
    return format_code(format) == format_code(static_cast<std::uint32_t>(DataFormat::Float32)) ||
           format_code(format) == format_code(static_cast<std::uint32_t>(DataFormat::Int32));
}

constexpr bool is_bfp_a_format(const std::uint32_t format)
{
    return format_code(format) == static_cast<std::uint32_t>(DataFormat::Bfp8) || format_code(format) == static_cast<std::uint32_t>(DataFormat::Bfp4) ||
           format_code(format) == static_cast<std::uint32_t>(DataFormat::Bfp2);
}

// Index (0/1) of the Dest half currently owned by this thread in SyncHalf mode.
// Math and pack each keep their own copy, flipped at the end of every Dest section.
inline std::uint32_t dest_half_index = 0;

inline std::uint32_t dest_half_base()
{
    return dest_half_index ? DEST_HALF_ROWS : 0;
}

} // namespace hal_llk
