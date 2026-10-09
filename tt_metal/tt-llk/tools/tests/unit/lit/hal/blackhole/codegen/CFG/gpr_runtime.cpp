// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -d %t.assert.o | FileCheck %s --check-prefix=ASSERT

// Each HAL function is compared with its adjacent reference_ function.
// Disable identical-code folding so both bodies must be emitted independently.

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// hal::gpr(index) selects the GPR at runtime, so the transfer is built in a
// register and pushed through the instruction buffer with the index in bits
// 23:16 (WRCFG). RDCFG accepts only hal::gpr<Index>().

inline constexpr cfg::Field state_last_block {cfg::RegisterScope::State, 32, 220, 0, 0, 32, 1, 0};

// WRCFG 32-bit to word 76; Deferred emits no trailing NOP.
extern "C" __attribute__((noinline, used)) void write_runtime_gpr_default_completion(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr(index));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_gpr_default_completion(std::uint32_t index)
{
    TT_WRCFG(index, 0, 76);
}

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_wait(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr(index));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_gpr_wait(std::uint32_t index)
{
    TT_WRCFG(index, 0, 76);
    TTI_NOP;
}

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_last_block(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, state_last_block, cfg::Sec::S0, cfg::GprTransferSize::Bits128, cfg::WrcfgCompletion::Deferred>(hal::gpr(index));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_gpr_last_block(std::uint32_t index)
{
    TT_WRCFG(index, 1, 220);
}

// The runtime transfer is last in the group, so Wait appends the final NOP.
extern "C" __attribute__((noinline, used)) void write_runtime_from_gpr_wait_group(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 1>(),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr(index)));
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_from_gpr_wait_group(std::uint32_t index)
{
    TTI_RMWCIB0(0x0f, 1, 0);
    TTI_RMWCIB0(0x01, 1, 5);
    TT_WRCFG(index, 0, 76);
    TTI_NOP;
}

extern "C" __attribute__((noinline, used)) void write_runtime_from_gpr_deferred_group(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr(index)),
        cfg::set<cfg::Thcon[cfg::Reg4].Base_cntx4_address, cfg::Sec::S0, 3>());
}

extern "C" __attribute__((noinline, used)) void reference_write_runtime_from_gpr_deferred_group(std::uint32_t index)
{
    TT_WRCFG(index, 1, 76);
    TTI_RMWCIB0(0xff, 3, 80);
    TTI_RMWCIB1(0xff, 0, 80);
    TTI_RMWCIB2(0xff, 0, 80);
    TTI_RMWCIB3(0xff, 0, 80);
}

#ifdef ENABLE_LLK_ASSERT

// Known invalid values use the runtime operand factory so their checks must
// emit ebreak. Keep these separate from release instruction-encoding checks.
extern "C" void write_runtime_gpr_out_of_range()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr(64));
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr(256));
}

// ASSERT-LABEL: <write_runtime_gpr_out_of_range>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_gpr_misaligned()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr(5)));
}

// ASSERT-LABEL: <write_runtime_gpr_misaligned>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_gpr_valid_boundaries()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr(63));
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr(60));
}

// ASSERT-LABEL: <write_runtime_gpr_valid_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret

#endif
