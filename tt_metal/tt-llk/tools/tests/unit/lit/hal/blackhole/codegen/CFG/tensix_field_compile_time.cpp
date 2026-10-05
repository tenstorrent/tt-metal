// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/cfg.h"

namespace cfg = hal::cfg;

extern "C" __attribute__((noinline, used)) void write_state_byte_0()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0xa>();
}

// CHECK-LABEL: <write_state_byte_0>:
// CHECK-NEXT: ttrmwcib0 15,10,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_byte_1()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg::Dstacc_val, cfg::Sec::S0, 0xa>();
}

// CHECK-LABEL: <write_state_byte_1>:
// CHECK-NEXT: ttrmwcib1 60,40,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_byte_2()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0, 1>();
}

// CHECK-LABEL: <write_state_byte_2>:
// CHECK-NEXT: ttrmwcib2 1,1,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_byte_3()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>();
}

// CHECK-LABEL: <write_state_byte_3>:
// CHECK-NEXT: ttrmwcib3 32,32,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_byte_straddle()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0xb>();
}

// SrcB_val occupies bits 8:5, so it touches byte lanes 0 and 1.
// CHECK-LABEL: <write_state_byte_straddle>:
// CHECK-NEXT: ttrmwcib0 224,96,0
// CHECK-NEXT: ttrmwcib1 1,1,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_full_word()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0, 0x12345678>();
}

// CHECK-LABEL: <write_state_full_word>:
// CHECK-NEXT: ttrmwcib0 255,120,186
// CHECK-NEXT: ttrmwcib1 255,86,186
// CHECK-NEXT: ttrmwcib2 255,52,186
// CHECK-NEXT: ttrmwcib3 255,18,186
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_state_section()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1, 2>();
}

// CHECK-LABEL: <write_state_section>:
// CHECK-NEXT: ttrmwcib0 15,2,112
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thread_section()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S7, 0x2a>();
}

// CHECK-LABEL: <write_thread_section>:
// CHECK-NEXT: ttsetc16 19,10752
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_one_constant_assignment()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 5>());
}

// CHECK-LABEL: <write_one_constant_assignment>:
// CHECK-NEXT: ttrmwcib0 15,5,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_same_word_constant_group()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0, 3>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>());
}

// CHECK-LABEL: <write_same_word_constant_group>:
// CHECK-NEXT: ttrmwcib0 31,19,64
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_straddling_constant_group()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 5>(), cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 7>());
}

// The merged mask 0x1ef spans two byte lanes.
// CHECK-LABEL: <write_straddling_constant_group>:
// CHECK-NEXT: ttrmwcib0 239,229,0
// CHECK-NEXT: ttrmwcib1 1,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_interleaved_constant_groups()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, 1>());
}

// Word 1 is emitted once at its first occurrence, before word 5.
// CHECK-LABEL: <write_interleaved_constant_groups>:
// CHECK-NEXT: ttrmwcib3 96,96,1
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thread_constant_group()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S2, 3>(), cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S2, 1>());
}

// CHECK-LABEL: <write_thread_constant_group>:
// CHECK-NEXT: ttsetc16 14,67
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_same_field_two_sections()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S1, 2>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0, 3>(),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S1, 1>());
}

// Sections resolve to different words, so they never share a group.
// CHECK-LABEL: <write_same_field_two_sections>:
// CHECK-NEXT: ttrmwcib0 31,18,112
// CHECK-NEXT: ttrmwcib0 15,3,64
// CHECK-NEXT: ret
