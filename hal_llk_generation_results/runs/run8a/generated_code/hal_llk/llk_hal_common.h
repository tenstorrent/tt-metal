// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the HAL-based LLK implementation (Blackhole).
// Everything in this directory is built only on the public HAL interface
// (hal/cfg.h, hal/sync.h, hal/unpack.h, hal/fpu.h, hal/pack.h, hal/mop.h, ...).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_memory_checks.h" // L1_ADDRESS
#include "tensor_shape.h"

// Like the LLK headers these replace, expose the ckernel names (DataFormat, BroadcastType, ...) to the kernel.
using namespace ckernel;

// Per-thread software state shared with the test harness (defined in the test source).
extern std::uint32_t unp_cfg_context;
extern std::uint32_t pack_sync_tile_dst_ptr;
extern std::uint32_t math_sync_tile_dst_index;

namespace hal_llk
{

// Semaphore roles. The indices match ckernel::semaphore so that kernels mixing these
// functions with other LLK code agree on the handshakes.
inline constexpr hal::sync::Semaphore MATH_PACK_SEM   = hal::sync::Semaphore::S1; // math <-> pack on dest
inline constexpr hal::sync::Semaphore UNPACK_SYNC_SEM = hal::sync::Semaphore::S5; // trisc <-> unpacker on cfg contexts

// Number of unpacker configuration contexts used for double buffering the per-tile L1 addresses.
inline constexpr std::uint32_t UNPACK_NUM_CONTEXTS = 2;

// Dest geometry (16-bit rows).
inline constexpr std::uint32_t DEST_HALF_ROWS     = DEST_REGISTER_FULL_SIZE / 2;
inline constexpr std::uint32_t DEST_ROWS_PER_TILE = 64; // 32x32 tile slot: 4 faces x 16 rows
inline constexpr std::uint32_t FPU_ROWS           = 8;  // rows consumed by one ELWADD/ELWSUB/ELWMUL

// SrcA write address (in datums) of unpacker 0. Math row 0 of SrcA (FPU reads and MOVD2A rows)
// is unpacker address 4 * 16: found on hardware, with 0 every eltwise result is wrong, with 64
// they are right (ckernel_instr_params.h nevertheless gives p_mova2d::MATH_HALO_ROWS = 0).
inline constexpr std::uint32_t SRCA_UNPACK_START = 4 * FACE_C_DIM;

// Hardware data-format code (low 4 bits of the DataFormat value).
constexpr std::uint32_t hw_format(const std::uint32_t format)
{
    return format & 0xF;
}

// Datum size in bytes keyed on the format family (low 2 bits): Float32/Tf32/Int32 -> 4,
// Float16/Float16_b/UInt16 -> 2, everything else -> 1.
constexpr std::uint32_t datum_bytes(const std::uint32_t format)
{
    return ((format & 0x3) == 0) ? 4 : (((format & 0x3) == 1) ? 2 : 1);
}

constexpr bool is_int8_or_int32(const std::uint32_t format)
{
    return (hw_format(format) == hw_format(static_cast<std::uint32_t>(DataFormat::Int8))) ||
           (hw_format(format) == hw_format(static_cast<std::uint32_t>(DataFormat::Int32)));
}

constexpr bool is_32bit_format(const std::uint32_t format)
{
    return (format == static_cast<std::uint32_t>(DataFormat::Int32)) || (format == static_cast<std::uint32_t>(DataFormat::UInt32)) ||
           (format == static_cast<std::uint32_t>(DataFormat::Float32));
}

// Base row of the dest half currently owned by this thread.
inline std::uint32_t dest_section_base()
{
    return (ckernel::dest_offset_id != 0) ? DEST_HALF_ROWS : 0;
}

} // namespace hal_llk

