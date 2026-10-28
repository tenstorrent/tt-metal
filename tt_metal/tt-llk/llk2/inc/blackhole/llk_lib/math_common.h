// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, math thread, Blackhole: the parts every FPU op's configure shares. Written only in HAL calls
// (tt_llk_blackhole/llk_lib/hal/), except where noted. The op headers (matmul.h, eltwise_binary.h) call these from
// their <op>_configure; nothing here runs on its own.
//
//   configure_math_formats<DestFp32, A, B>()   Dest mode, the ALU format baseline and the Src zero flag, after
//                                               waiting for math and SFPU to drain
//   set_addr_mod<Slot, A, B, D, Fidelity>()    programs one address modifier (ADDR_MOD) slot
//
// Address modifiers move the math row counters (RWCs) after each FPU instruction that names them. The counters
// follow the ISA's ApplyAddrMod (tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor/RWCs.md): a plain
// step adds incr to the counter; a carriage-return step adds incr to the counter's CR register and copies it to the
// counter; clear zeroes both.
//
// Not HAL yet: the Src zero-flag rule (requires_disabled_src_zero_flag, cmath_common.h) and the ckernel caches of it.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "tensix_types.h" // DataFormat

namespace llk2::llk_lib
{

/// Row counter behavior of one address modifier: {increment, clear, carriage return} for SrcA, SrcB or Dest.
struct counter_step
{
    std::uint32_t incr; ///< rows the counter (or its CR register, when cr is set) moves on
    std::uint32_t clr;  ///< 1: counter and CR to 0
    std::uint32_t cr;   ///< 1: add incr to the CR register and copy it to the counter
};

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
        cfg::set<AddrMod[cfg::Fidelity].Clear, Slot, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, AddrMod[cfg::Bias].Incr, Slot, 0>();
}

/// Dest mode, ALU format baseline and Src zero flag for an FPU op whose SrcA holds SrcAFormat and SrcB SrcBFormat
/// (the register formats). The math side of today's hw_configure. Waits for math and SFPU to drain first.
template <bool DestFp32, DataFormat SrcAFormat, DataFormat SrcBFormat>
inline __attribute__((always_inline)) void configure_math_formats()
{
    namespace cfg = hal::cfg;
    using namespace ckernel;
    using hal::sync::StallCondition;
    using hal::sync::StallTarget;
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
    constexpr std::uint32_t src_a     = static_cast<std::uint32_t>(SrcAFormat);
    constexpr std::uint32_t src_b     = static_cast<std::uint32_t>(SrcBFormat);
    constexpr std::uint32_t int8_math = is_int8_or_int32_format(src_a) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, int8_math>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, DestFp32 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, DestFp32 ? 1u : 0u>();
    // Src formats come from what the unpackers wrote (Blackhole's implied formats), not an override.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());

    // Src zero-flag baseline for these formats; the ckernel caches stay in step for the LLK asserts.
    math::src_zero_flag_srca_fmt  = src_a;
    math::src_zero_flag_srcb_fmt  = src_b;
    const std::uint32_t zero_flag = requires_disabled_src_zero_flag(src_a, src_b) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(zero_flag);
    math::src_zero_flag_hw = zero_flag;
}

} // namespace llk2::llk_lib
