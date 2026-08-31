// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
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

extern "C" __attribute__((noinline, used)) void write_thread_section()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S7, 0x2a>();
}

// CHECK-LABEL: <write_thread_section>:
// CHECK-NEXT: ttsetc16 19,10752
// CHECK-NEXT: ret
