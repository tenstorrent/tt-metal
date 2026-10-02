// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

// Runtime operands are added to a constant operation word and pushed through the
// instruction buffer. Out-of-range runtime operands branch to an LLK assertion.

// SETDMAREG is 0x45000000 | payload << 8 | half index; GPR 5 owns halves 10 and 11.
extern "C" __attribute__((noinline, used)) void set_runtime_word(std::uint32_t value)
{
    gpr_ops::set(hal::gpr<5>(), value);
}

// CHECK-LABEL: <set_runtime_word>:
// CHECK-NOT: ebreak
// CHECK: lui [[SET:a[0-7]]],0x45000
// CHECK-NOT: ebreak
// CHECK: addi {{a[0-7]}},[[SET]],10
// CHECK-NOT: ebreak
// CHECK: srli a0,a0,0x10
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: addi {{a[0-7]}},[[SET]],11
// CHECK: sw {{a[0-7]}},0([[BUF:a[0-7]]])
// CHECK-NEXT: sw {{a[0-7]}},0([[BUF]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_runtime_low(std::uint32_t value)
{
    gpr_ops::set_low(hal::gpr<5>(), value);
}

// CHECK-LABEL: <set_runtime_low>:
// CHECK-NEXT: lui [[LIMIT:a[0-7]]],0x10
// CHECK-NEXT: bgeu a0,[[LIMIT]],[[FAIL:[0-9a-f]+]]
// CHECK: lui [[SET:a[0-7]]],0x45000
// CHECK-NEXT: addi [[OP:a[0-7]]],[[SET]],10
// CHECK-NEXT: slli a0,a0,0x8
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void set_runtime_high(std::uint32_t value)
{
    gpr_ops::set_high(hal::gpr<5>(), value);
}

// CHECK-LABEL: <set_runtime_high>:
// CHECK-NEXT: lui [[LIMIT:a[0-7]]],0x10
// CHECK-NEXT: bgeu a0,[[LIMIT]],[[FAIL:[0-9a-f]+]]
// CHECK: lui [[SET:a[0-7]]],0x45000
// CHECK-NEXT: addi [[OP:a[0-7]]],[[SET]],11
// CHECK-NEXT: slli a0,a0,0x8
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ADDDMAREG is 0x58000000 | rhs_is_immediate << 23 | dst << 12 | rhs << 6 | lhs.
extern "C" __attribute__((noinline, used)) void add_runtime_immediate(std::uint32_t value)
{
    gpr_ops::add(hal::gpr<1>(), hal::gpr<2>(), gpr_ops::immediate(value));
}

// CHECK-LABEL: <add_runtime_immediate>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[ADD:a[0-7]]],0x58801
// CHECK-NEXT: addi [[OP:a[0-7]]],[[ADD]],2
// CHECK-NEXT: slli a0,a0,0x6
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_gpr(std::uint32_t index)
{
    gpr_ops::add(hal::gpr<1>(), hal::gpr<2>(), hal::gpr(index));
}

// CHECK-LABEL: <add_runtime_gpr>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[ADD:a[0-7]]],0x58001
// CHECK-NEXT: addi [[OP:a[0-7]]],[[ADD]],2
// CHECK-NEXT: slli a0,a0,0x6
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// CMPDMAREG Equal is 0x5d000000 | 1 << 23 | 2 << 18 | dst << 12 | 3 << 6 | 2.
extern "C" __attribute__((noinline, used)) void compare_runtime_destination(std::uint32_t index)
{
    gpr_ops::compare<gpr_ops::Compare::Equal>(hal::gpr(index), hal::gpr<2>(), gpr_ops::immediate<3>());
}

// CHECK-LABEL: <compare_runtime_destination>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[CMP:a[0-7]]],0x5d880
// CHECK-NEXT: addi [[OP:a[0-7]]],[[CMP]],194
// CHECK-NEXT: slli a0,a0,0xc
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak
