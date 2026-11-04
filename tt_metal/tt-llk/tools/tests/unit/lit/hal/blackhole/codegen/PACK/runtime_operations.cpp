// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -c %t/register_write_value.cpp -o %t/register_write_value.o
// RUN: %{blackhole_objdump} -dr %t/register_write_value.o | FileCheck %s --enable-var-scope
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// codegen.cpp is compiled without LLK_ASSERT so runtime paths contain only the
// issued instruction; asserts.cpp checks the runtime operand asserts.
// register_write_value.cpp keeps FileCheck because no macro names the
// PACR_SETREG load-flavor value field, and composing it from the write-flavor
// macro folds the opcode into one constant while the HAL adds it last.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/pack.h"

namespace pack = hal::pack;

//--- codegen.cpp
#include "common.h"

// Runtime descriptors are encoded in a register and pushed through the
// instruction buffer.

extern "C" __attribute__((noinline, used)) void run_runtime_interfaces(const std::uint8_t interfaces)
{
    pack::run(pack::DataTransfer {.interfaces = interfaces, .boundary = pack::TileBoundary::Last});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_interfaces(const std::uint8_t interfaces)
{
    TT_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        0 /*DstAccessMode*/,
        0 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        interfaces,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        1 /*Last*/);
}

extern "C" __attribute__((noinline, used)) void run_runtime_register_write(const std::uint8_t stream_id)
{
    pack::run(pack::RegisterWrite {.address_slot = 2, .stream_id = stream_id});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_register_write(const std::uint8_t stream_id)
{
    TT_PACR_SETREG(1 /*Push*/, 0 /*ModeSel*/, 0 /*Unused*/, 0 /*DisableStall*/, 2 /*AddrSel*/, stream_id, 1 /*Flush*/, 0 /*Last*/);
}

extern "C" __attribute__((noinline, used)) void run_runtime_edge_window(const std::uint8_t x_end)
{
    pack::run(pack::EdgeWindow {.x_end = x_end});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_edge_window(const std::uint8_t x_end)
{
    TT_SETPKEDGOF(0 /*y_end*/, 0 /*y_start*/, x_end, 0 /*x_start*/);
}

//--- register_write_value.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void run_runtime_register_write_value(const std::uint16_t value)
{
    pack::run(pack::RegisterWriteValue {.high_half = true, .value = value});
}

// Load flavor: opcode 0x4a, ModeSel at bit 22, the value at bits 18:3 and the
// high-half select at bit 2. Objdump comments the addi and sw, so those lines
// end at the comment marker instead of {{$}}.
// CHECK-LABEL: <run_runtime_register_write_value>:
// CHECK-DAG: lui [[FLAGS:a[0-7]]],0x400{{$}}
// CHECK-DAG: addi [[FLAGS]],[[FLAGS]],4 #
// CHECK-DAG: slli [[FIELD:a[0-7]]],a0,0x3{{$}}
// CHECK-DAG: or [[FIELDS:a[0-7]]],[[FIELD]],[[FLAGS]]{{$}}
// CHECK-DAG: lui [[OPCODE:a[0-7]]],0x4a000{{$}}
// CHECK-DAG: add [[WORD:a[0-7]]],[[FIELDS]],[[OPCODE]]{{$}}
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer{{$}}
// CHECK: sw [[WORD]],0({{a[0-7]}}) #
// CHECK-NEXT: R_RISCV_LO12_S __instrn_buffer{{$}}
// CHECK-NEXT: R_RISCV_RELAX *ABS*{{$}}
// CHECK-NEXT: ret

//--- asserts.cpp
#include "common.h"

// Invalid runtime operands must trap before the instruction is issued; valid
// boundaries must not trap.

extern "C" void check_runtime_interfaces(const std::uint8_t interfaces)
{
    pack::run(pack::DataTransfer {.interfaces = interfaces, .boundary = pack::TileBoundary::Last});
}

// ASSERT-LABEL: <check_runtime_interfaces>:
// ASSERT: li [[LIMIT:a[0-7]]],15{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_interfaces()
{
    pack::run(pack::DataTransfer {.interfaces = 16});
}

// ASSERT-LABEL: <reject_invalid_interfaces>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_register_write(const std::uint8_t stream_id)
{
    pack::run(pack::RegisterWrite {.address_slot = 2, .stream_id = stream_id});
}

// ASSERT-LABEL: <check_runtime_register_write>:
// ASSERT: li [[LIMIT:a[0-7]]],63{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_stream_id()
{
    pack::run(pack::RegisterWrite {.stream_id = 64});
}

// ASSERT-LABEL: <reject_invalid_stream_id>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_edge_window(const std::uint8_t x_end)
{
    pack::run(pack::EdgeWindow {.x_end = x_end});
}

// ASSERT-LABEL: <check_runtime_edge_window>:
// ASSERT: li [[LIMIT:a[0-7]]],15{{$}}
// ASSERT-NEXT: bgeu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_edge_window()
{
    pack::run(pack::EdgeWindow {.x_end = 16});
}

// ASSERT-LABEL: <reject_invalid_edge_window>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

// Every staged half is encodable, so a runtime value adds no assert.
extern "C" void accept_runtime_register_write_value(const std::uint16_t value)
{
    pack::run(pack::RegisterWriteValue {.high_half = true, .value = value});
}

// ASSERT-LABEL: <accept_runtime_register_write_value>:
// ASSERT-NOT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_boundaries()
{
    pack::run(pack::DataTransfer {.interfaces = 15});
    pack::run(pack::RegisterWrite {.stream_id = 63});
    pack::run(pack::EdgeWindow {.x_end = 15});
}

// ASSERT-LABEL: <accept_runtime_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
