// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// The last state word (223) and the last THCON block (176-179) hold no named
// field, so these descriptors reach the bank ends directly.
inline constexpr cfg::Field state_last {cfg::RegisterScope::State, 32, 223, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_last_block {cfg::RegisterScope::State, 32, 220, 0, 0, 32, 1, 0};
inline constexpr cfg::Field thcon_last_block {cfg::RegisterScope::State, 32, 176, 0, 0, 32, 1, 0};

// WRCFG through Access::TensixCfgUnit.

extern "C" __attribute__((noinline, used)) void write_gpr_default_completion()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(hal::gpr<5>());
}

// Completion defaults to Deferred: no NOP unless Wait is requested.
// CHECK-LABEL: <write_gpr_default_completion>:
// CHECK-NEXT: ttwrcfg 5,0,77
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_gpr_wait()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<4>());
}

// CHECK-LABEL: <write_gpr_wait>:
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_gpr_deferred()
{
    cfg::
        write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<5>());
}

// CHECK-LABEL: <write_gpr_deferred>:
// CHECK-NEXT: ttwrcfg 5,0,77
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_gpr_128_default_completion()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1, cfg::GprTransferSize::Bits128>(hal::gpr<16>());
}

// CHECK-LABEL: <write_gpr_128_default_completion>:
// CHECK-NEXT: ttwrcfg 16,1,112
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_gpr_128_wait()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1, cfg::GprTransferSize::Bits128, cfg::WrcfgCompletion::Wait>(
        hal::gpr<16>());
}

// CHECK-LABEL: <write_gpr_128_wait>:
// CHECK-NEXT: ttwrcfg 16,1,112
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_reset_via_gpr()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::StateReset::EN, cfg::Sec::S0>(hal::gpr<4>());
}

// A field-granular Tensix write to the state-reset register lowers to
// RMWCIB, which hardware ignores; a whole-word GPR transfer reaches it.
// CHECK-LABEL: <write_state_reset_via_gpr>:
// CHECK-NEXT: ttwrcfg 4,0,4
// CHECK-NEXT: ret

// A field group stands in for its Raw anchor in whole-word GPR transfers.
extern "C" __attribute__((noinline, used)) void write_field_group_gpr()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

// CHECK-LABEL: <write_field_group_gpr>:
// CHECK-NEXT: ttwrcfg 4,1,112
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_field_group_from_gpr()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

// CHECK-LABEL: <write_field_group_from_gpr>:
// CHECK-NEXT: ttwrcfg 4,1,112
// CHECK-NEXT: ret

// GPR transfers grouped with field assignments.

extern "C" __attribute__((noinline, used)) void write_ordered_constant_operations()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<4>()),
        cfg::from_gpr<cfg::Thcon[cfg::Reg4].Base_cntx4_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr<5>()),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 2>());
}

// The two word-0 fields merge across both GPR transfers at their first occurrence.
// Only the Wait transfer is followed by a NOP.
// CHECK-LABEL: <write_ordered_constant_operations>:
// CHECK-NEXT: ttrmwcib0 239,65,0
// CHECK-NEXT: ttrmwcib1 1,0,0
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ttwrcfg 5,0,80
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_deferred_gpr_sequence()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<4>()),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>());
}

// CHECK-LABEL: <write_deferred_gpr_sequence>:
// CHECK-NEXT: ttrmwcib0 15,1,0
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thread_constants_across_gpr()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 2>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()),
        cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());
}

// CHECK-LABEL: <write_thread_constants_across_gpr>:
// CHECK-NEXT: ttsetc16 5,6
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_word_after_gpr_transfer()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()),
        cfg::set<cfg::Thcon[cfg::Reg4].Base_cntx4_address, cfg::Sec::S0, 3>());
}

// Word 80 immediately follows the four-word transfer to 76-79 and is disjoint.
// CHECK-LABEL: <write_word_after_gpr_transfer>:
// CHECK-NEXT: ttwrcfg 4,1,76
// CHECK-NEXT: ttrmwcib0 255,3,80
// CHECK-NEXT: ttrmwcib1 255,0,80
// CHECK-NEXT: ttrmwcib2 255,0,80
// CHECK-NEXT: ttrmwcib3 255,0,80
// CHECK-NEXT: ret

// REG2FLOP through Access::TensixScalarUnit: flop index = word - 64, size 1 = 32-bit.

extern "C" __attribute__((noinline, used)) void write_thcon_gpr_scalar()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>());
}

// CHECK-LABEL: <write_thcon_gpr_scalar>:
// CHECK-NEXT: ttreg2flop 1,0,0,0,12,4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thcon_gpr_scalar_last_index()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<63>());
}

// CHECK-LABEL: <write_thcon_gpr_scalar_last_index>:
// CHECK-NEXT: ttreg2flop 1,0,0,0,12,63
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thcon_gpr_128_scalar()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1, cfg::GprTransferSize::Bits128>(hal::gpr<16>());
}

// CHECK-LABEL: <write_thcon_gpr_128_scalar>:
// CHECK-NEXT: ttreg2flop 0,0,0,0,48,16
// CHECK-NEXT: ret

// RDCFG copies the complete word that contains the field.

extern "C" __attribute__((noinline, used)) void read_cfg_to_gpr()
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr<5>());
}

// CHECK-LABEL: <read_cfg_to_gpr>:
// CHECK-NEXT: ttrdcfg 5,186
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void read_cfg_field_word_to_gpr()
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1>(hal::gpr<6>());
}

// CHECK-LABEL: <read_cfg_field_word_to_gpr>:
// CHECK-NEXT: ttrdcfg 6,112
// CHECK-NEXT: ret

// Bank-end boundaries.

extern "C" __attribute__((noinline, used)) void write_gpr_last_word()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_last, cfg::Sec::S0>(hal::gpr<4>());
}

// CHECK-LABEL: <write_gpr_last_word>:
// CHECK-NEXT: ttwrcfg 4,0,223
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_grouped_gpr_last_word()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_last, cfg::Sec::S0>(hal::gpr<4>()));
}

// CHECK-LABEL: <write_grouped_gpr_last_word>:
// CHECK-NEXT: ttwrcfg 4,0,223
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_gpr_last_block()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_last_block, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

// CHECK-LABEL: <write_gpr_last_block>:
// CHECK-NEXT: ttwrcfg 4,1,220
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_grouped_gpr_last_block()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_last_block, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

// CHECK-LABEL: <write_grouped_gpr_last_block>:
// CHECK-NEXT: ttwrcfg 4,1,220
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void read_gpr_last_word()
{
    cfg::read<cfg::Access::TensixCfgUnit, state_last, cfg::Sec::S0>(hal::gpr<4>());
}

// CHECK-LABEL: <read_gpr_last_word>:
// CHECK-NEXT: ttrdcfg 4,223
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_scalar_last_thcon_block()
{
    cfg::write<cfg::Access::TensixScalarUnit, thcon_last_block, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

// CHECK-LABEL: <write_scalar_last_thcon_block>:
// CHECK-NEXT: ttreg2flop 0,0,0,0,112,4
// CHECK-NEXT: ret
