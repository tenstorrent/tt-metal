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
#include "hal/mop.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h" // L1_ADDRESS
#include "tensor_shape.h"

// The LLK API is used unqualified by its callers.
using namespace ckernel;

namespace hal_llk
{

namespace cfg  = hal::cfg;
namespace sync = hal::sync;

// Semaphore roles shared by the unpack, math and pack threads.
constexpr sync::Semaphore SEM_MATH_PACK = sync::Semaphore::S1; // math <-> pack handoff of a Dest section

constexpr std::uint32_t DEST_ROWS_PER_TILE = 64; // rows of one 32x32 tile in Dest
constexpr std::uint32_t DEST_ROWS_PER_FACE = 16;

constexpr std::uint32_t fmt(const DataFormat format)
{
    return static_cast<std::uint32_t>(format);
}

// Register-file datum size in bytes, derived from the low two format bits (Float32/Tf32 -> 4, Float16* -> 2, else 1).
constexpr std::uint32_t register_datum_bytes(const std::uint32_t format)
{
    return ((format & 0x3) == fmt(DataFormat::Float32)) ? 4 : ((format & 0x3) == fmt(DataFormat::Float16)) ? 2 : 1;
}

constexpr bool is_int8_math_format(const std::uint32_t format)
{
    return ((format & 0xF) == fmt(DataFormat::Int8)) || ((format & 0xF) == fmt(DataFormat::Int32));
}

// Base row of the Dest section currently owned by this thread.
inline std::uint32_t dest_section_base()
{
    return (ckernel::dest_offset_id != 0) ? DEST_REGISTER_HALF_SIZE : 0;
}

} // namespace hal_llk
