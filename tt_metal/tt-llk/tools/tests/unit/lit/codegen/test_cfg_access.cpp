// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: blackhole-cfg, sfpi
// DEFINE: %{compile} = %{sfpi_cxx} %{cfg_flags} -fno-ipa-icf -S %s
// RUN: %{compile} -O3 -DCOMPILE_FOR_TRISC=0 -o %t.t0.s
// RUN: %{python} %S/../Inputs/compare_cfg_asm.py %t.t0.s
// RUN: FileCheck %s --check-prefix=MMIO < %t.t0.s
// RUN: %{compile} -O3 -DCOMPILE_FOR_TRISC=2 -o %t.t2.s
// RUN: %{python} %S/../Inputs/compare_cfg_asm.py %t.t2.s
// RUN: FileCheck %s --check-prefix=MMIO < %t.t2.s

#include <cstdint>

#include "../Inputs/cfg_test_fields.h"

// Each actual/expected pair must generate identical instructions. References
// spell out the hardware operations rather than using the CFG backend.
// Use -O3 so the small MMIO bank selectors are inlined into both sides.

// Hardware layout: two banks of 224 uint32_t words, selected by cfg_state_id.
inline volatile std::uint32_t tt_reg_ptr* expected_state_bank()
{
    const std::uint32_t word_offset = ckernel::cfg_state_id == 0 ? 0u : 224u;
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE) + word_offset;
}

// Two runtime setters and two constant setters are shuffled across a GPR
// transfer. Merge by word; emit groups in order of their first occurrence.
extern "C" void actual_shuffled(std::uint32_t format, std::uint32_t uncompressed)
{
    write<Access::TensixCfgUnit>(
        set<ThconTileDescriptorFields::InDataFormat, Sec::S0>(format),
        set<ThconReg2Fields::Out_data_format, Sec::S0, 5>(),
        from_gpr<ThconReg3Fields::Base_address, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>()),
        set<ThconTileDescriptorFields::Uncompressed, Sec::S0>(uncompressed),
        set<ThconReg2Fields::Throttle_mode, Sec::S0, 1>());
}

extern "C" void expected_shuffled(std::uint32_t format, std::uint32_t uncompressed)
{
    TT_RMWCIB0(0x1f, (format & 0xf) | ((uncompressed & 1u) << 4), 64);
    TTI_RMWCIB0(0x3f, 0x15, 72);
    TTI_WRCFG(4, 1, 76);
    TTI_NOP;
}

// A runtime setter must retain constant bits from the same word.
extern "C" void actual_mixed_word(std::uint32_t format)
{
    write<Access::TensixCfgUnit>(
        set<ThconTileDescriptorFields::Uncompressed, Sec::S0, 1>(),
        from_gpr<ThconReg3Fields::Base_address, Sec::S0>(hal::gpr<4>()),
        set<ThconTileDescriptorFields::InDataFormat, Sec::S0>(format));
}

extern "C" void expected_mixed_word(std::uint32_t format)
{
    TT_RMWCIB0(0x1f, (format & 0xf) | 0x10, 64);
    TTI_WRCFG(4, 0, 76);
    TTI_NOP;
}

// State word 5 and Thread word 5 are separate groups. Thread fields separated
// by another scope and a GPR transfer must still emit a single SETC16.
extern "C" void actual_thread_runtime(std::uint32_t base, std::uint32_t override_address)
{
    write<Access::TensixCfgUnit>(
        set<SrcASet::Base, Sec::S0>(base),
        set<state_word5, Sec::S0, 5>(),
        from_gpr<ThconReg3Fields::Base_address, Sec::S0>(hal::gpr<4>()),
        set<SrcASet::SetOvrdWithAddr, Sec::S0>(override_address));
}

extern "C" void expected_thread_runtime(std::uint32_t base, std::uint32_t override_address)
{
    TT_SETC16(5, (base & 3u) | ((override_address & 1u) << 2));
    TTI_RMWCIB0(0xf, 5, 5);
    TTI_WRCFG(4, 0, 76);
    TTI_NOP;
}

extern "C" void actual_thread_constant()
{
    write<Access::TensixCfgUnit>(
        set<SrcASet::Base, Sec::S0, 2>(), from_gpr<ThconReg3Fields::Base_address, Sec::S0>(hal::gpr<4>()), set<SrcASet::SetOvrdWithAddr, Sec::S0, 1>());
}

extern "C" void expected_thread_constant()
{
    TTI_SETC16(5, 6);
    TTI_WRCFG(4, 0, 76);
    TTI_NOP;
}

// MMIO expression reassociation can change register allocation relative to a
// handwritten reference. Check the combined mask/data and single store directly.
// MMIO-LABEL: mmio_group:
// MMIO-DAG: lw [[OLD:[a-z0-9]+]],256([[CFG:[a-z0-9]+]])
// MMIO-DAG: andi a0,a0,15
// MMIO: andi [[OLD]],[[OLD]],-32
// MMIO: or [[DATA:[a-z0-9]+]],{{[a-z0-9]+}},{{[a-z0-9]+}}
// MMIO: ori [[DATA]],[[DATA]],16
// MMIO: sw [[DATA]],256([[CFG]])
// MMIO-NOT: sw
// MMIO: ret
extern "C" void mmio_group(std::uint32_t format)
{
    write<Access::MMIO>(set<ThconTileDescriptorFields::InDataFormat, Sec::S0>(format), set<ThconTileDescriptorFields::Uncompressed, Sec::S0, 1>());
}

// Sections resolve to distinct addresses before grouping.
extern "C" void actual_sections()
{
    write<Access::TensixCfgUnit>(
        set<ThconTileDescriptorFields::InDataFormat, Sec::S1, 2>(),
        set<ThconTileDescriptorFields::InDataFormat, Sec::S0, 3>(),
        set<ThconTileDescriptorFields::Uncompressed, Sec::S1, 1>());
}

extern "C" void expected_sections()
{
    TTI_RMWCIB0(0x1f, 0x12, 112);
    TTI_RMWCIB0(0x0f, 3, 64);
}

// The word immediately after a four-word transfer is disjoint.
inline constexpr Field after_transfer {RegisterScope::State, 32, 80, 0, 0, 4, 1, 0};

extern "C" void actual_adjacent()
{
    write<Access::TensixCfgUnit>(from_gpr<ThconReg3Fields::Base_address, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>()), set<after_transfer, Sec::S0, 3>());
}

extern "C" void expected_adjacent()
{
    TTI_WRCFG(4, 1, 76);
    TTI_NOP;
    TTI_RMWCIB0(0xf, 3, 80);
}

// Bank-boundary successes complement the compile-fail diagnostics.
extern "C" void actual_gpr_last()
{
    write<Access::TensixCfgUnit, state_last, Sec::S0>(hal::gpr<4>());
}

extern "C" void expected_gpr_last()
{
    TTI_WRCFG(4, 0, 223);
    TTI_NOP;
}

extern "C" void actual_grouped_gpr_last()
{
    write<Access::TensixCfgUnit>(from_gpr<state_last, Sec::S0>(hal::gpr<4>()));
}

extern "C" void expected_grouped_gpr_last()
{
    TTI_WRCFG(4, 0, 223);
    TTI_NOP;
}

extern "C" void actual_gpr_last_block()
{
    write<Access::TensixCfgUnit, state_last_block, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
}

extern "C" void expected_gpr_last_block()
{
    TTI_WRCFG(4, 1, 220);
    TTI_NOP;
}

extern "C" void actual_grouped_gpr_last_block()
{
    write<Access::TensixCfgUnit>(from_gpr<state_last_block, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>()));
}

extern "C" void expected_grouped_gpr_last_block()
{
    TTI_WRCFG(4, 1, 220);
    TTI_NOP;
}

extern "C" void actual_dynamic_deferred(std::uint32_t index)
{
    write<Access::TensixCfgUnit, state_last_block, Sec::S0, GprTransferSize::Bits128, WrcfgCompletion::Deferred>(hal::gpr(index));
}

extern "C" void expected_dynamic_deferred(std::uint32_t index)
{
    TT_WRCFG(index, 1, 220);
}

extern "C" void actual_gpr_read()
{
    read<Access::TensixCfgUnit, state_last, Sec::S0>(hal::gpr<4>());
}

extern "C" void expected_gpr_read()
{
    TTI_RDCFG(4, 223);
}

extern "C" void actual_scalar_last_block()
{
    write<Access::TensixScalarUnit, thcon_last_block, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
}

extern "C" void expected_scalar_last_block()
{
    TTI_REG2FLOP(0, 0, 0, 0, 112, 4);
}

extern "C" void actual_array_last(const std::array<std::uint32_t, 2>& values)
{
    write<Access::MMIO, ChickenBits::sfpu_scbd_disable, Sec::S0, 2>(values);
}

extern "C" void expected_array_last(const std::array<std::uint32_t, 2>& values)
{
    auto cfg = expected_state_bank();
    cfg[222] = values[0];
    cfg[223] = values[1];
}

extern "C" void actual_array_full_bank(const std::array<std::uint32_t, 224>& values)
{
    write<Access::MMIO, state_first, Sec::S0, 224>(values);
}

extern "C" void expected_array_full_bank(const std::array<std::uint32_t, 224>& values)
{
    auto cfg = expected_state_bank();
    for (std::uint32_t i = 0; i < 224; ++i)
    {
        cfg[i] = values[i];
    }
}

extern "C" std::uint32_t actual_state_last()
{
    return read_word<Access::MMIO, PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, Sec::S0, 183>();
}

extern "C" std::uint32_t expected_state_last()
{
    return expected_state_bank()[223];
}

extern "C" std::uint32_t actual_thread_last()
{
    return read_word<Access::MMIO, SrcASet::Base, Sec::S0, 62>();
}

extern "C" std::uint32_t expected_thread_last()
{
    ckernel::reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, 672 + COMPILE_FOR_TRISC * 68 + 67);
    ckernel::wait(1);
    return ckernel::reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA) & 0xffffu;
}

extern "C" std::uint32_t actual_thread_field()
{
    return read<Access::MMIO, SrcASet::Base, Sec::S0, ThreadTarget::T2>();
}

extern "C" std::uint32_t expected_thread_field()
{
    ckernel::reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, 672 + 2 * 68 + 5);
    ckernel::wait(1);
    return ckernel::reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA) & 3u;
}
