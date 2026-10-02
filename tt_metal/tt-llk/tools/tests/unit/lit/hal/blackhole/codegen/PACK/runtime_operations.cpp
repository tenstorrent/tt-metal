// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/pack.h"

namespace pack = hal::pack;

// Runtime descriptors are encoded in a register and pushed through the
// instruction buffer; out-of-range fields reach the LLK assertion first.

extern "C" __attribute__((noinline, used)) void run_runtime_interfaces(std::uint8_t interfaces)
{
    pack::run(pack::DataTransfer {.interfaces = interfaces, .boundary = pack::TileBoundary::Last});
}

// CHECK-LABEL: <run_runtime_interfaces>:
// CHECK: li [[MAX:a[0-7]]],15
// CHECK-NEXT: bltu [[MAX]],a0,{{[0-9a-f]+}} <[[ASSERT:.L[0-9]+]]>
// CHECK-DAG: lui [[BASE:a[0-7]]],0x41000
// CHECK-DAG: addi [[OP:a[0-7]]],[[BASE]],1
// CHECK-DAG: slli [[FIELD:a[0-7]]],a0,0x8
// CHECK-DAG: add [[WORD:a[0-7]]],[[FIELD]],[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK: <[[ASSERT]]>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void run_runtime_register_write(std::uint8_t stream_id)
{
    pack::run(pack::RegisterWrite {.address_slot = 2, .stream_id = stream_id});
}

// CHECK-LABEL: <run_runtime_register_write>:
// CHECK: li [[MAX:a[0-7]]],63
// CHECK-NEXT: bltu [[MAX]],a0,{{[0-9a-f]+}} <[[ASSERT:.L[0-9]+]]>
// CHECK-DAG: lui [[OP:a[0-7]]],0x4a800
// CHECK-DAG: addi [[OP]],[[OP]],514
// CHECK-DAG: sh2add [[WORD:a[0-7]]],a0,[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK: <[[ASSERT]]>:
// CHECK-NEXT: ebreak

// The staged half has no invalid encoding, so no assertion is emitted.
extern "C" __attribute__((noinline, used)) void run_runtime_register_write_value(std::uint16_t value)
{
    pack::run(pack::RegisterWriteValue {.high_half = true, .value = value});
}

// CHECK-LABEL: <run_runtime_register_write_value>:
// CHECK-NOT: ebreak
// CHECK-DAG: lui [[FLAGS:a[0-7]]],0x400
// CHECK-DAG: addi [[FLAGS]],[[FLAGS]],4
// CHECK-DAG: slli [[FIELD:a[0-7]]],a0,0x3
// CHECK-DAG: or [[FIELDS:a[0-7]]],[[FIELD]],[[FLAGS]]
// CHECK-DAG: lui [[OPCODE:a[0-7]]],0x4a000
// CHECK-DAG: add [[WORD:a[0-7]]],[[FIELDS]],[[OPCODE]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK-NOT: ebreak
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_runtime_edge_window(std::uint8_t x_end)
{
    pack::run(pack::EdgeWindow {.x_end = x_end});
}

// CHECK-LABEL: <run_runtime_edge_window>:
// CHECK: li [[MAX:a[0-7]]],15
// CHECK-NEXT: bgeu [[MAX]],a0,{{[0-9a-f]+}} <[[VALID:.L[0-9]+]]>
// CHECK: ebreak
// CHECK-EMPTY:
// CHECK-NEXT: <[[VALID]]>:
// CHECK-DAG: slli [[FIELD:a[0-7]]],a0,0x4
// CHECK-DAG: lui [[OP:a[0-7]]],0x1d000
// CHECK-DAG: add [[WORD:a[0-7]]],[[FIELD]],[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
