// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the HAL-based LLK implementations in tests/hal_llk/.
// Only pure format arithmetic and thin wrappers around hal:: interfaces live here.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_memory_checks.h" // L1_ADDRESS(): byte address -> LLK 16-byte address convention
#include "tensix_types.h"

// The LLK entry points are declared at global scope and name ckernel types unqualified (as the spec does).
using namespace ckernel;

namespace hal_llk
{

// ---------------------------------------------------------------------------
// Data-format helpers (pure arithmetic on the DataFormat codes, no hardware access)
// ---------------------------------------------------------------------------

constexpr std::uint32_t fmt(const DataFormat f)
{
    return static_cast<std::uint32_t>(f);
}

constexpr std::uint32_t format_code(const std::uint32_t format)
{
    return format & 0xF; // the hardware format fields are four bits wide
}

constexpr bool is_bfp_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1F;
    return f == fmt(DataFormat::Bfp8) || f == fmt(DataFormat::Bfp8_b) || f == fmt(DataFormat::Bfp4) || f == fmt(DataFormat::Bfp4_b) ||
           f == fmt(DataFormat::Bfp2) || f == fmt(DataFormat::Bfp2_b);
}

constexpr bool is_bfp_a_format(const std::uint32_t format)
{
    const std::uint32_t f = format & 0x1F;
    return f == fmt(DataFormat::Bfp8) || f == fmt(DataFormat::Bfp4) || f == fmt(DataFormat::Bfp2);
}

constexpr bool is_int8_or_int32(const std::uint32_t format)
{
    return format_code(format) == fmt(DataFormat::Int8) || format == fmt(DataFormat::Int32);
}

constexpr bool is_integer_format(const std::uint32_t format)
{
    return format == fmt(DataFormat::Int8) || format == fmt(DataFormat::UInt8) || format == fmt(DataFormat::Int32) || format == fmt(DataFormat::UInt32) ||
           format == fmt(DataFormat::UInt16);
}

constexpr bool is_32bit_format(const std::uint32_t format)
{
    return format == fmt(DataFormat::Float32) || format == fmt(DataFormat::Int32) || format == fmt(DataFormat::UInt32);
}

// Bytes occupied by one datum in a register file: Float32/Tf32 -> 4, Float16/Float16_b -> 2, else 1.
constexpr std::uint32_t register_datum_bytes(const std::uint32_t format)
{
    return ((format & 0x3) == fmt(DataFormat::Float32)) ? 4 : ((format & 0x3) == fmt(DataFormat::Float16)) ? 2 : 1;
}

// Dest rows per 32x32 tile slot and per DEST half, shared by the math and pack threads.
constexpr std::uint32_t DEST_ROWS_PER_TILE = 64;
constexpr std::uint32_t DEST_HALF_ROWS     = 512;

// ---------------------------------------------------------------------------
// Math (FPU) address modifiers: one hal::cfg grouped write per ADDR_MOD slot.
// Every field of the slot is written so stale state from an earlier kernel cannot leak in.
// ---------------------------------------------------------------------------

struct MathAddrMod
{
    std::uint8_t srca_incr  = 0;
    bool srca_cr            = false;
    bool srca_clear         = false;
    std::uint8_t srcb_incr  = 0;
    bool srcb_cr            = false;
    bool srcb_clear         = false;
    std::uint16_t dest_incr = 0;
    bool dest_cr            = false;
    bool dest_clear         = false;
    bool dest_c_to_cr       = false;
    std::uint8_t fid_incr   = 0;
    bool fid_clear          = false;
};

template <hal::cfg::Sec Slot>
inline void program_math_addr_mod(const MathAddrMod m)
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, Slot>(m.srca_incr & 0x3F),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, Slot>(m.srca_cr),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, Slot>(m.srca_clear),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, Slot>(m.srcb_incr & 0x3F),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, Slot>(m.srcb_cr),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, Slot>(m.srcb_clear),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, Slot>((m.srca_incr >> 6) & 0x1),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr2, Slot>((m.srcb_incr >> 6) & 0x1),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, Slot>(m.dest_incr),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, Slot>(m.dest_cr),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, Slot>(m.dest_clear),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, Slot>(m.dest_c_to_cr),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, Slot>(m.fid_incr),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, Slot>(m.fid_clear),
        cfg::set<cfg::AddrMod[cfg::Bias].Incr, Slot>(0),
        cfg::set<cfg::AddrMod[cfg::Bias].Clear, Slot>(0));
}

} // namespace hal_llk
