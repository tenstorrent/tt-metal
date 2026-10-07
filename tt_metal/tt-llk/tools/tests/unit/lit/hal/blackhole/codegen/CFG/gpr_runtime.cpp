// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -d %t.assert.o | FileCheck %s --check-prefix=ASSERT

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// hal::gpr(index) selects the GPR at runtime, so the transfer is built in a
// register and pushed through the instruction buffer with the index in bits
// 23:16 (WRCFG) or 5:0 (REG2FLOP). RDCFG accepts only hal::gpr<Index>().

inline constexpr cfg::Field state_last_block {cfg::RegisterScope::State, 32, 220, 0, 0, 32, 1, 0};

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_default_completion(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr(index));
}

// WRCFG 32-bit to word 76 is 0xb0000000 + 76; Deferred emits no trailing NOP.
// CHECK-LABEL: <write_runtime_gpr_default_completion>:
// CHECK-DAG: lui [[OP:a[0-7]]],0xb0000
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],76
// CHECK-DAG: slli a0,a0,0x10
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ret

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

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_wait(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr(index));
}

// CHECK-LABEL: <write_runtime_gpr_wait>:
// CHECK-DAG: lui [[OP:a[0-7]]],0xb0000
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],76
// CHECK-DAG: slli a0,a0,0x10
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_last_block(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit, state_last_block, cfg::Sec::S0, cfg::GprTransferSize::Bits128, cfg::WrcfgCompletion::Deferred>(hal::gpr(index));
}

// CHECK-LABEL: <write_runtime_gpr_last_block>:
// CHECK-DAG: lui [[OP:a[0-7]]],0xb0008
// CHECK-DAG: addi {{a[0-7]}},[[OP]],220
// CHECK-DAG: slli a0,a0,0x10
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_from_gpr_wait_group(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr(index)),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>());
}

// The runtime transfer keeps its position between the two constant words and
// its NOP precedes the next field write.
// CHECK-LABEL: <write_runtime_from_gpr_wait_group>:
// CHECK-NEXT: ttrmwcib0 15,1,0
// CHECK-DAG: lui [[OP:a[0-7]]],0xb0000
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],76
// CHECK-DAG: slli a0,a0,0x10
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_from_gpr_deferred_group(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr(index)),
        cfg::set<cfg::Thcon[cfg::Reg4].Base_cntx4_address, cfg::Sec::S0, 3>());
}

// CHECK-LABEL: <write_runtime_from_gpr_deferred_group>:
// CHECK-DAG: lui [[OP:a[0-7]]],0xb0008
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],76
// CHECK-DAG: slli a0,a0,0x10
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ttrmwcib0 255,3,80
// CHECK-NEXT: ttrmwcib1 255,0,80
// CHECK-NEXT: ttrmwcib2 255,0,80
// CHECK-NEXT: ttrmwcib3 255,0,80
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_scalar(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr(index));
}

// REG2FLOP 32-bit to flop 12 is 0x48400000 + (12 << 6).
// CHECK-LABEL: <write_runtime_gpr_scalar>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x48400
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],768
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_gpr_128_scalar(std::uint32_t index)
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, cfg::GprTransferSize::Bits128>(hal::gpr(index));
}

// REG2FLOP 128-bit to flop 48 is 0x48000000 + (48 << 6).
// CHECK-LABEL: <write_runtime_gpr_128_scalar>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x48001
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],-1024
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ret
