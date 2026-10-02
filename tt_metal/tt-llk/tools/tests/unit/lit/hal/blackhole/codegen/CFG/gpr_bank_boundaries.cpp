// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// The last state word (223) and the last THCON block (176-179) hold no named
// field, so these descriptors reach the bank ends directly.
inline constexpr cfg::Field state_last {cfg::RegisterScope::State, 32, 223, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_last_block {cfg::RegisterScope::State, 32, 220, 0, 0, 32, 1, 0};
inline constexpr cfg::Field thcon_last_block {cfg::RegisterScope::State, 32, 176, 0, 0, 32, 1, 0};

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

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_last_block(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, state_last_block, cfg::Sec::S0, cfg::GprTransferSize::Bits128, cfg::WrcfgCompletion::Deferred>(hal::gpr(index));
}

// The WRCFG word is 0xb0008000 (128-bit) + 220, with the runtime GPR index in
// bits 23:16. Deferred completion emits no trailing NOP.
// CHECK-LABEL: <write_runtime_gpr_last_block>:
// CHECK-DAG: lui [[OP:a[0-7]]],0xb0008
// CHECK-DAG: addi {{a[0-7]}},[[OP]],220
// CHECK-DAG: slli a0,a0,0x10
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
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
