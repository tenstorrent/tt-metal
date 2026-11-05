// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, Blackhole: the parts every math op's configure shares. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/), except where noted. The op headers (matmul.h, eltwise_binary.h, copy.h, vec.h)
// call these from their <op>_configure and <op>_execute; nothing here runs on its own.
//
//   configure_math_formats<DestFp32, A, B>()   after math and the SFPU drain: Dest mode, the ALU format baseline,
//                                               implied SrcA format, Src bank release and the Src zero flag
//   set_addr_mod<Slot, A, B, D, Fidelity>()    programs one address modifier (ADDR_MOD) slot
//   reset_math_counters()                      every row counter (and its CR register) and the fidelity phase to 0
//   dest_rows_per_tile(params)                 Dest rows one tile of an op takes
//
// Address modifiers move the math row counters (RWCs) after each FPU instruction that names them. The counters
// follow the ISA's ApplyAddrMod (tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/RWCs.md): a plain
// step adds incr to the counter; a carriage-return (CR) step adds incr to the counter's CR register and copies it to
// the counter; clear zeroes both, whatever the CR bit says.
//
// Not HAL yet: the Src zero-flag rule (requires_disabled_src_zero_flag, cmath_common.h) and the ckernel caches of
// it, and is_int8_or_int32_format (ckernel_defs.h).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "tensix_types.h" // DataFormat

namespace llk2::llk_lib
{

/// Rows of SrcB and Dest one MVMUL covers, and rows of SrcA, SrcB and Dest one ELWADD, ELWSUB or MOVA2D (its
/// eight-row form) covers.
inline constexpr std::uint32_t rows_per_fpu_op = 8;

/// The first replay slot the math thread records one tile into: ckernel::math::replay_buf_offset, which leaves
/// slots [0, 16) to SFPU code.
inline constexpr std::uint32_t math_replay_tile_start = 16;

/// Row counter behavior of one address modifier: {increment, clear, carriage return} for SrcA, SrcB or Dest.
struct counter_step
{
    std::uint32_t incr; ///< rows the counter (or its CR register, when cr is set) moves on
    std::uint32_t clr;  ///< 1: counter and CR register to 0; cr is then ignored (ApplyAddrMod)
    std::uint32_t cr;   ///< 1: add incr to the CR register and copy it to the counter
};

/// The counter does not move.
inline constexpr counter_step counter_stay {0, 0, 0};

/// The counter and its CR register to 0. The CR bit is left 0: clear ignores it.
inline constexpr counter_step counter_clear {0, 1, 0};

/// Dest rows one tile of an op takes: faces per tile times rows per face. P is any math params struct with
/// num_faces and face_r_dim.
template <class P>
constexpr std::uint32_t dest_rows_per_tile(const P& p)
{
    return p.num_faces * p.face_r_dim;
}

/// Points this thread's math Dest offset at Dest row `row` (absolute): the FPU's and the SFPU's Dest addresses on this
/// thread count from there.
inline __attribute__((always_inline)) void set_math_dest_row(std::uint32_t row)
{
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DestTargetRegCfgMath::Offset, hal::cfg::Sec::S0>(row);
}

/// Points this thread's math Dest offset at tile `tile` of the tensor that starts at Dest row dest_row (P's tile size).
template <auto P>
inline __attribute__((always_inline)) void set_math_dest_tile(std::uint32_t dest_row, std::uint32_t tile)
{
    set_math_dest_row(dest_row + tile * dest_rows_per_tile(P));
}

/// Programs address modifier Slot. Fields this does not name are written 0, as addr_mod_t does.
template <hal::cfg::Sec Slot, counter_step A, counter_step B, counter_step D, std::uint32_t FidelityIncr = 0>
inline __attribute__((always_inline)) void set_addr_mod()
{
    namespace cfg = hal::cfg;
    using cfg::AddrMod;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::SrcA].Incr, Slot, A.incr>(),
        cfg::set<AddrMod[cfg::SrcA].Clear, Slot, A.clr>(),
        cfg::set<AddrMod[cfg::SrcA].CR, Slot, A.cr>(),
        cfg::set<AddrMod[cfg::SrcB].Incr, Slot, B.incr>(),
        cfg::set<AddrMod[cfg::SrcB].Clear, Slot, B.clr>(),
        cfg::set<AddrMod[cfg::SrcB].CR, Slot, B.cr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::Dest].Incr, Slot, D.incr>(),
        cfg::set<AddrMod[cfg::Dest].Clear, Slot, D.clr>(),
        cfg::set<AddrMod[cfg::Dest].CR, Slot, D.cr>(),
        cfg::set<AddrMod[cfg::Fidelity].Incr, Slot, FidelityIncr>(),
        cfg::set<AddrMod[cfg::Fidelity].Clear, Slot, 0>());                     // the fidelity phase only moves on (HiFi) or stays
    cfg::write<cfg::Access::TensixCfgUnit, AddrMod[cfg::Bias].Incr, Slot, 0>(); // no bias counter step
}

/// Every math row counter (SrcA, SrcB, Dest), their CR registers, and the fidelity phase to 0.
inline __attribute__((always_inline)) void reset_math_counters()
{
    using namespace hal::counters::rwc;
    set<MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .clear_fidelity = true}>();
}

/// Dest mode, ALU format baseline, implied SrcA format, Src bank release and Src zero flag for an FPU op whose SrcA
/// holds SrcAFormat and SrcB SrcBFormat (the register formats). The math side of today's hw_configure. Waits for
/// math and SFPU to drain first.
template <bool DestFp32, DataFormat SrcAFormat, DataFormat SrcBFormat>
inline __attribute__((always_inline)) void configure_math_formats()
{
    namespace cfg = hal::cfg;
    using namespace ckernel;
    using hal::sync::StallCondition;
    using hal::sync::StallTarget;
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    // ZEROACC indices are relative to the Dest offset, not absolute.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
    constexpr std::uint32_t src_a     = static_cast<std::uint32_t>(SrcAFormat);
    constexpr std::uint32_t src_b     = static_cast<std::uint32_t>(SrcBFormat);
    constexpr std::uint32_t int8_math = is_int8_or_int32_format(src_a) ? 1u : 0u; // not HAL: ckernel_defs.h
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, int8_math>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, DestFp32 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, DestFp32 ? 1u : 0u>();
    // Src formats come from what the unpackers wrote (Blackhole's implied formats), not an override.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());
    // SrcA's format is the one implied by the bank the unpacker filled, and a Src release (a flip or an
    // instruction's release) hands the bank back to the unpacker: it clears its data valid. Each is one whole
    // thread-config word.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    // Src zero-flag baseline for these formats; the ckernel caches stay in step for the LLK asserts.
    math::src_zero_flag_srca_fmt  = src_a;
    math::src_zero_flag_srcb_fmt  = src_b;
    const std::uint32_t zero_flag = requires_disabled_src_zero_flag(src_a, src_b) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(zero_flag);
    math::src_zero_flag_hw = zero_flag;
}

} // namespace llk2::llk_lib
