// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --check-prefix=OPS

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using L1ToL1 = move::Transfer<move::L1, move::L1>;

// THCON mover 0 words: 88 = source, 89 = destination, 90 = size | direction 3 << 30.

constexpr L1ToL1 copy {
    .source      = {.block = 0x10000},
    .destination = {.block = 0},
    .size        = {.value = 0x8000},
};

constexpr L1ToL1 copy_wait {
    .source      = {.block = 0x100},
    .destination = {.block = 0x200},
    .size        = {.value = 16},
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void l1_to_l1_issue_only()
{
    move::run<copy>();
}

// CHECK-LABEL: <l1_to_l1_issue_only>:
// CHECK-NEXT: ttrmwcib0 255,0,88
// CHECK-NEXT: ttrmwcib1 255,0,88
// CHECK-NEXT: ttrmwcib2 255,1,88
// CHECK-NEXT: ttrmwcib3 255,0,88
// CHECK-NEXT: ttrmwcib0 255,0,89
// CHECK-NEXT: ttrmwcib1 255,0,89
// CHECK-NEXT: ttrmwcib2 255,0,89
// CHECK-NEXT: ttrmwcib3 255,0,89
// CHECK-NEXT: ttrmwcib0 255,0,90
// CHECK-NEXT: ttrmwcib1 255,128,90
// CHECK-NEXT: ttrmwcib2 255,0,90
// CHECK-NEXT: ttrmwcib3 255,192,90
// CHECK-NEXT: ttxmov 0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void l1_to_l1_wait()
{
    move::run<copy_wait>();
}

// CHECK-LABEL: <l1_to_l1_wait>:
// CHECK-NEXT: ttrmwcib0 255,0,88
// CHECK-NEXT: ttrmwcib1 255,1,88
// CHECK-NEXT: ttrmwcib2 255,0,88
// CHECK-NEXT: ttrmwcib3 255,0,88
// CHECK-NEXT: ttrmwcib0 255,0,89
// CHECK-NEXT: ttrmwcib1 255,2,89
// CHECK-NEXT: ttrmwcib2 255,0,89
// CHECK-NEXT: ttrmwcib3 255,0,89
// CHECK-NEXT: ttrmwcib0 255,16,90
// CHECK-NEXT: ttrmwcib1 255,0,90
// CHECK-NEXT: ttrmwcib2 255,0,90
// CHECK-NEXT: ttrmwcib3 255,192,90
// CHECK-NEXT: ttxmov 0,0
// CHECK-NEXT: ttstallwait 511,512
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void l1_to_l1_runtime(
    const std::uint32_t source, const std::uint32_t destination, const std::uint32_t size, const move::Completion completion)
{
    move::run(L1ToL1 {.source = {.block = source}, .destination = {.block = destination}, .size = {.value = size}, .completion = completion});
}

// Runtime words are split into RMWCIB byte lanes: twelve instruction-buffer
// writes, then XMOV, then the MoverIdle wait only for Completion::Wait.
// CHECK-LABEL: <l1_to_l1_runtime>:
// CHECK-COUNT-12: sw {{[at][0-7]}},0({{[at][0-7]}})
// CHECK-NEXT: lui [[R0:a[0-7]]],0x40000
// CHECK-NEXT: sw [[R0]],0([[BUF:a[0-7]]])
// CHECK-NEXT: li [[R0]],1
// CHECK-NEXT: beq a3,[[R0]],{{[0-9a-f]+}}
// CHECK-NEXT: R_RISCV_BRANCH
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: <.L{{[0-9]+}}>:
// CHECK-NEXT: lui [[R0]],0xa2ff8
// CHECK-NEXT: addi [[R0]],[[R0]],512
// CHECK-NEXT: sw [[R0]],0([[BUF]])
// CHECK-NEXT: ret

// The RMWCIB0..3 opcode bases for words 88..90, and direction 3 merged into word 90.
// OPS-LABEL: <l1_to_l1_runtime>:
// OPS-DAG: lui [[CIB0:[at][0-7]]],0xb3ff0
// OPS-DAG: lui [[CIB1:[at][0-7]]],0xb4ff0
// OPS-DAG: lui [[CIB2:[at][0-7]]],0xb5ff0
// OPS-DAG: lui [[CIB3:[at][0-7]]],0xb6ff0
// OPS-DAG: addi {{[at][0-7]}},[[CIB0]],88
// OPS-DAG: addi {{[at][0-7]}},[[CIB3]],89
// OPS-DAG: addi {{[at][0-7]}},[[CIB2]],90
// OPS-DAG: lui {{[at][0-7]}},0xc0000
