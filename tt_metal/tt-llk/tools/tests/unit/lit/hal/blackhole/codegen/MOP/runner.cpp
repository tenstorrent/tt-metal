// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

using Template0Runner = mop::Runner<mop::MopTemplate::Template0>;
using Template1Runner = mop::Runner<mop::MopTemplate::Template1>;

// MOP operands are (template, count - 1 or outer-count high bits, mask low 16 bits
// or packed counts). MOP_CFG carries the upper 16 mask bits.

extern "C" __attribute__((noinline, used)) void run_template0_short()
{
    Template0Runner::run<4, 0x5>();
}

// A mask of at most 16 bits applied to at most 16 iterations needs no MOP_CFG.
// CHECK-LABEL: <run_template0_short>:
// CHECK-NEXT: ttmop 0,3,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_template0_long_count()
{
    Template0Runner::run<128, 0x5>();
}

// CHECK-LABEL: <run_template0_long_count>:
// CHECK-NEXT: ttmopcfg 0
// CHECK-NEXT: ttmop 0,127,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_template0_wide_mask()
{
    Template0Runner::run<1, 0x12345>();
}

// CHECK-LABEL: <run_template0_wide_mask>:
// CHECK-NEXT: ttmopcfg 1
// CHECK-NEXT: ttmop 0,0,9029
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_template0_runtime_count(std::uint32_t count)
{
    Template0Runner::run<0x12345>(count);
}

// MOP_CFG is immediate; the MOP word 0x01002345 | (count - 1) << 16 goes through the instruction buffer.
// CHECK-LABEL: <run_template0_runtime_count>:
// CHECK-NEXT: addi [[ENCODED:a[0-7]]],a0,-1
// CHECK-NEXT: li [[LIMIT:a[0-7]]],127
// CHECK-NEXT: bltu [[LIMIT]],[[ENCODED]],{{[0-9a-f]+}} <[[INVALID:\.L[0-9]+]]>
// CHECK-NOT: ttmop
// CHECK: ttmopcfg 1
// CHECK-NOT: ttmop
// CHECK: lui [[OP:a[0-7]]],0x1002
// CHECK: slli [[COUNT:a[0-7]]],[[ENCODED]],0x10
// CHECK: addi [[OP]],[[OP]],837
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add [[WORD:a[0-7]]],[[COUNT]],[[OP]]
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: <[[INVALID]]>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void run_template0_runtime_mask(std::uint32_t mask)
{
    Template0Runner::run_with_runtime_mask<4>(mask);
}

// MOP_CFG (0x03000000 | mask >> 16) is pushed before MOP (0x01030000 | mask & 0xffff).
// CHECK-LABEL: <run_template0_runtime_mask>:
// CHECK-NOT: ebreak
// CHECK: srli [[HI:a[0-7]]],a0,0x10
// CHECK: zext.h [[LO:a[0-7]]],a0
// CHECK: lui [[CFGOP:a[0-7]]],0x3000
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add [[CFGWORD:a[0-7]]],[[HI]],[[CFGOP]]
// CHECK: lui [[MOPOP:a[0-7]]],0x1030
// CHECK: sw [[CFGWORD]],0([[BUF:a[0-7]]])
// CHECK: add [[MOPWORD:a[0-7]]],[[LO]],[[MOPOP]]
// CHECK: sw [[MOPWORD]],0([[BUF]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_template0_runtime(std::uint32_t count, std::uint32_t mask)
{
    Template0Runner::run(count, mask);
}

// CHECK-LABEL: <run_template0_runtime>:
// CHECK-NEXT: addi [[ENCODED:a[0-7]]],a0,-1
// CHECK-NEXT: li [[LIMIT:a[0-7]]],127
// CHECK-NEXT: bltu [[LIMIT]],[[ENCODED]],{{[0-9a-f]+}} <[[INVALID:\.L[0-9]+]]>
// CHECK-NOT: ttmop
// CHECK: srli [[HI:a[0-7]]],a1,0x10
// CHECK: lui [[CFGOP:a[0-7]]],0x3000
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add [[CFGWORD:a[0-7]]],[[HI]],[[CFGOP]]
// CHECK: sw [[CFGWORD]],0([[BUF:a[0-7]]])
// CHECK-NEXT: sw {{a[0-7]}},0([[BUF]])
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: <[[INVALID]]>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void run_template1()
{
    Template1Runner::run();
}

// CHECK-LABEL: <run_template1>:
// CHECK-NEXT: ttmop 1,0,0
// CHECK-NEXT: ret

inline constexpr mop::Template1CountOverrides overrides {.outer_loop_count = 0x3ff, .inner_loop_count = 5};

extern "C" __attribute__((noinline, used)) void run_template1_overrides()
{
    Template1Runner::run<overrides>();
}

// Outer bits 9:6 fill the count operand; outer bits 5:0 sit above the 10-bit inner count.
// CHECK-LABEL: <run_template1_overrides>:
// CHECK-NEXT: ttmop 1,15,64517
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_template1_runtime_overrides(std::uint32_t outer, std::uint32_t inner)
{
    Template1Runner::run({.outer_loop_count = outer, .inner_loop_count = inner});
}

// CHECK-LABEL: <run_template1_runtime_overrides>:
// CHECK-NEXT: or [[BOTH:a[0-7]]],a0,a1
// CHECK-NEXT: li [[LIMIT:a[0-7]]],1023
// CHECK-NEXT: bgeu [[LIMIT]],[[BOTH]],
// CHECK: ebreak
// CHECK-NOT: ttmop
// CHECK: andi {{a[0-7]}},a1,1023
// CHECK: andi {{a[0-7]}},a0,960
// CHECK: lui [[OP:a[0-7]]],0x1800
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t get_template1_operation()
{
    return mop::get_operation<overrides>();
}

// 0x018ffc05: the same operation as run_template1_overrides, returned instead of issued.
// CHECK-LABEL: <get_template1_operation>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0x1900
// CHECK-NEXT: addi a0,[[OP]],-1019
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t get_template1_runtime_operation(std::uint32_t outer, std::uint32_t inner)
{
    return mop::get_operation({.outer_loop_count = outer, .inner_loop_count = inner});
}

// CHECK-LABEL: <get_template1_runtime_operation>:
// CHECK: ebreak
// CHECK-NOT: __instrn_buffer
// CHECK: andi {{a[0-7]}},a1,1023
// CHECK: andi {{a[0-7]}},a0,960
// CHECK: lui [[OP:a[0-7]]],0x1800
// CHECK-NOT: __instrn_buffer
// CHECK: ret
