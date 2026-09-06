// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/cfg.h"

namespace cfg = hal::cfg;

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
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 5>(), cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 7>());
}

// CHECK-LABEL: <write_same_word_constant_group>:
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

extern "C" __attribute__((noinline, used)) void write_ordered_constant_operations()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(cfg::gpr<4, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>()),
        cfg::from_gpr<cfg::Thcon[cfg::Reg4].Base_cntx4_address, cfg::Sec::S0>(hal::gpr<5>()),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 2>());
}

// CHECK-LABEL: <write_ordered_constant_operations>:
// CHECK-NEXT: ttrmwcib0 15,1,0
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ttwrcfg 5,0,80
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ttrmwcib0 224,64,0
// CHECK-NEXT: ttrmwcib1 1,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_constant_batch()
{
    cfg::write<cfg::Access::TensixCfgUnit>([](auto& out) { out(cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 5>()); });
}

// CHECK-LABEL: <write_constant_batch>:
// CHECK-NEXT: ttrmwcib0 15,5,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_constant_operations_batch()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        [](auto& out)
        {
            out(cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 1>(),
                cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(cfg::gpr<4, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>()),
                cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>());
        });
}

// CHECK-LABEL: <write_constant_operations_batch>:
// CHECK-NEXT: ttrmwcib0 15,1,0
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ret
