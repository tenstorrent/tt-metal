// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared state and helpers for the HAL-based LLK implementation (Blackhole).
// Every LLK function in tests/hal_llk/ is written against the HAL interface only.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/gpr_ops.h"
#include "hal/math_counters.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/pack.h"
#include "hal/sync.h"
#include "hal/unpack.h"
#include "hal/utils/address_modifier.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "tensor_shape.h"

namespace hal_llk
{

namespace cfg = hal::cfg;

// Semaphore roles (same numbering the rest of the firmware uses, see ckernel_structs.h).
constexpr hal::sync::Semaphore MATH_PACK_SEM   = hal::sync::Semaphore::S1; // math <-> pack on Dest
constexpr hal::sync::Semaphore UNPACK_SYNC_SEM = hal::sync::Semaphore::S5; // trisc <-> unpacker cfg contexts
constexpr hal::sync::Semaphore PACK_TILE_SEM   = hal::sync::Semaphore::S4; // trisc <-> packer L1 destination

// Number of unpacker configuration contexts used for per-tile double buffering of base addresses.
constexpr std::uint32_t NUM_UNPACK_CFG_CONTEXTS = 2;

// Dest geometry in 16-row face units. A Dest tile slot is always 4 faces (64 rows), whatever the tile shape.
constexpr std::uint32_t DEST_FACE_ROWS = 16;
constexpr std::uint32_t DEST_TILE_ROWS = 64;
constexpr std::uint32_t DEST_HALF_ROWS = 512;

// SrcA write address of unpacker 0 (in 4-datum units): starts past the 16 halo rows.
constexpr std::uint32_t UNPACK_SRCA_DEST_ADDRESS = 16 * 16 / 4;

// Per-thread software state. Each TRISC is a separate binary, so these are private to each thread.
inline std::uint32_t unpack_cfg_context = 0; // unpacker configuration context the next tile is written to
inline std::uint32_t math_dest_bank     = 0; // Dest half math is writing (SyncHalf)
inline std::uint32_t pack_dest_bank     = 0; // Dest half pack is reading (SyncHalf)

// Pack-thread GPRs (same allocation as ckernel_gpr_map.h p_gpr_pack).
constexpr std::uint32_t GPR_PACK_DEST_OFFSET_LO = 4;  // 4 GPRs: Dest offsets of the low half
constexpr std::uint32_t GPR_PACK_DEST_OFFSET_HI = 8;  // 4 GPRs: Dest offsets of the high half

constexpr std::uint32_t format_code(const std::uint32_t format)
{
    return format & 0xf;
}

// Datum size in a register file (Src/Dest) for a register format: Float32/Tf32/Int32 -> 4, Float16 family -> 2, else 1.
constexpr std::uint32_t register_datum_bytes(const std::uint32_t format)
{
    const std::uint32_t low = format & 0x3;
    return (low == static_cast<std::uint32_t>(DataFormat::Float32)) ? 4 : (low == static_cast<std::uint32_t>(DataFormat::Float16)) ? 2 : 1;
}

constexpr bool is_int8_math_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1f;
    return f == static_cast<std::uint32_t>(DataFormat::Int8) || f == static_cast<std::uint32_t>(DataFormat::Int32);
}

constexpr bool is_uint8_format(const std::uint32_t format)
{
    return (format & 0x1f) == static_cast<std::uint32_t>(DataFormat::UInt8);
}

constexpr bool is_32bit_register_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1f;
    return f == static_cast<std::uint32_t>(DataFormat::Float32) || f == static_cast<std::uint32_t>(DataFormat::Int32) ||
           f == static_cast<std::uint32_t>(DataFormat::UInt32) || f == static_cast<std::uint32_t>(DataFormat::Int8) ||
           f == static_cast<std::uint32_t>(DataFormat::UInt8);
}

} // namespace hal_llk
