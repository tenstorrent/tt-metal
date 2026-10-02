// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// A runtime value cannot be an immediate, so its RMWCIB/SETC16 is built in a
// register and pushed through the instruction buffer. The operation word holds
// the opcode, mask, and address; the masked runtime data lands in bits 15:8 or 15:0.

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
