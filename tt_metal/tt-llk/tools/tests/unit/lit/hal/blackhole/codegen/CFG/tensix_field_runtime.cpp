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

// A runtime value cannot be an immediate, so its RMWCIB/SETC16 is built in a
// register and pushed through the instruction buffer. The operation word holds
// the opcode, mask, and address; the runtime data lands in bits 15:8 (RMWCIB)
// or 15:0 (SETC16).

extern "C" __attribute__((noinline, used)) void write_runtime_state_byte_0(std::uint32_t format)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format);
}

// RMWCIB0, mask 0x0f, word 64.
// CHECK-LABEL: <write_runtime_state_byte_0>:
// CHECK-DAG: lui [[OP:a[0-7]]],0xb30f0
// CHECK-DAG: addi [[OPA:a[0-7]]],[[OP]],64
// CHECK-DAG: slli a0,a0,0x8
// CHECK-DAG: zext.h a0,a0
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OPA]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ret

#ifdef ENABLE_LLK_ASSERT

// Runtime assignments must validate values before masking, even when a group
// has only one member or the invalid assignment is not its first member.
extern "C" void write_runtime_group_single_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(2));
}

// ASSERT-LABEL: <write_runtime_group_single_overflow>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_group_later_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, 1>(), cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(2));
}

// ASSERT-LABEL: <write_runtime_group_later_overflow>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_thread_group_overflow()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0>(64));
}

// ASSERT-LABEL: <write_runtime_thread_group_overflow>:
// ASSERT-NOT: sw
// ASSERT: ebreak
// ASSERT: ret

extern "C" void write_runtime_group_valid_boundaries()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(0xffffffffu),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(1),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0>(63),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, cfg::Sec::S0>(1));
}

// ASSERT-LABEL: <write_runtime_group_valid_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret

#endif

extern "C" __attribute__((noinline, used)) void write_runtime_state_byte_straddle(std::uint32_t format)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>(format);
}

// SrcB_val (bits 8:5) needs RMWCIB0 with mask 0xe0 and RMWCIB1 with mask 0x01, both at word 0.
// CHECK-LABEL: <write_runtime_state_byte_straddle>:
// CHECK-DAG: slli [[LANE0:a[0-7]]],a0,0xd
// CHECK-DAG: lui [[OP0:a[0-7]]],0xb3e00
// CHECK-DAG: andi [[LANE1:a[0-7]]],a0,2040
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add [[LANE0]],[[LANE0]],[[OP0]]
// CHECK-DAG: slli [[LANE1]],[[LANE1]],0x5
// CHECK-DAG: lui [[OP1:a[0-7]]],0xb4010
// CHECK-DAG: sw [[LANE0]],0([[BUF:a[0-7]]])
// CHECK: add [[LANE1]],[[LANE1]],[[OP1]]
// CHECK-NEXT: sw [[LANE1]],0([[BUF]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_state_full_word(std::uint32_t seed)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(seed);
}

// Each byte lane gets its own RMWCIB at word 186.
// CHECK-LABEL: <write_runtime_state_full_word>:
// CHECK-DAG: lui [[OP0:a[0-7]]],0xb3ff0
// CHECK-DAG: lui [[OP1:a[0-7]]],0xb4ff0
// CHECK-DAG: lui [[OP2:a[0-7]]],0xb5ff0
// CHECK-DAG: lui [[OP3:a[0-7]]],0xb6ff0
// CHECK-DAG: addi {{a[0-7]}},[[OP0]],186
// CHECK-DAG: addi {{a[0-7]}},[[OP1]],186
// CHECK-DAG: addi {{a[0-7]}},[[OP2]],186
// CHECK-DAG: addi {{a[0-7]}},[[OP3]],186
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK-DAG: sw {{a[0-7]}},0([[BUF:a[0-7]]])
// CHECK-DAG: sw {{a[0-7]}},0([[BUF]])
// CHECK-DAG: sw {{a[0-7]}},0([[BUF]])
// CHECK-DAG: sw {{a[0-7]}},0([[BUF]])
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_runtime_thread_section(std::uint32_t increment)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AddrMod[cfg::SrcB].Incr, cfg::Sec::S7>(increment);
}

// SETC16 to thread word 19 with the value shifted to bits 13:8.
// CHECK-LABEL: <write_runtime_thread_section>:
// CHECK-DAG: slli a0,a0,0x8
// CHECK-DAG: lui [[OP:a[0-7]]],0xb2130
// CHECK-DAG: zext.h a0,a0
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_shuffled_runtime_group(std::uint32_t format, std::uint32_t uncompressed)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format),
        cfg::set<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0, 5>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0>(uncompressed),
        cfg::set<cfg::Thcon[cfg::Reg2].Throttle_mode, cfg::Sec::S0, 1>());
}

// Word 64 is emitted once with both runtime fields, before the later constant word.
// Only InDataFormat is masked; Uncompressed is the top field and the RMWCIB byte
// lane clips it.
// CHECK-LABEL: <write_shuffled_runtime_group>:
// CHECK-DAG: andi a0,a0,15
// CHECK-NOT: andi {{a[0-7]}},{{a[0-7]}},16
// CHECK-DAG: lui [[OP:a[0-7]]],0xb31f0
// CHECK-DAG: addi [[OP]],[[OP]],64
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ttrmwcib0 63,21,72
// CHECK-NEXT: ttwrcfg 4,1,76
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_mixed_constant_runtime_word(std::uint32_t format)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format));
}

// The constant Uncompressed bit (0x1000 = 1 << 4 << 8) joins the runtime format.
// CHECK-LABEL: <write_mixed_constant_runtime_word>:
// CHECK-DAG: andi a0,a0,15
// CHECK-DAG: lui {{a[0-7]}},0x1
// CHECK-DAG: lui [[OP:a[0-7]]],0xb31f0
// CHECK-DAG: addi [[OP]],[[OP]],64
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_thread_and_state_same_address(std::uint32_t base, std::uint32_t override_address)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::SrcASet::Base, cfg::Sec::S0>(base),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()),
        cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0>(override_address));
}

// Thread word 5 and state word 5 are separate: one SETC16 for both thread fields
// even across the GPR transfer, and one RMWCIB for the state field.
// SetOvrdWithAddr is the top field, so only the SETC16 group mask clips it.
// CHECK-LABEL: <write_thread_and_state_same_address>:
// CHECK-NOT: andi {{a[0-7]}},{{a[0-7]}},4
// CHECK: andi a0,a0,3
// CHECK-NOT: andi {{a[0-7]}},{{a[0-7]}},4
// CHECK: lui {{a[0-7]}},0xb2050
// CHECK-NOT: andi {{a[0-7]}},{{a[0-7]}},4
// CHECK: andi {{a[0-7]}},{{a[0-7]}},7
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ttrmwcib0 1,1,5
// CHECK-NEXT: ttwrcfg 4,0,76
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_runtime_fields_stacked(std::uint32_t mode, std::uint32_t threshold)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(mode), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(threshold));
}

// A runtime field is masked only when another field of its group lies above it:
// ApplyRelu (bits 5:2) keeps its mask, while ReluThreshold (bits 21:6) relies on
// the byte-lane masks and needs no 0x3fffc0 mask, which would be built with lui 0x400.
// CHECK-LABEL: <write_runtime_fields_stacked>:
// CHECK-NOT: lui {{a[0-7]}},0x400
// CHECK: andi {{a[0-7]}},{{a[0-7]}},60
// CHECK-NOT: lui {{a[0-7]}},0x400
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret
