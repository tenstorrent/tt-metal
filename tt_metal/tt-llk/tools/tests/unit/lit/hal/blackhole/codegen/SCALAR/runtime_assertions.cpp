// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// Check assertion branches separately from the TT_* encoding comparisons.
// These are compile-only checks; ebreak is inspected, not executed.

#include <cstdint>

#include "hal/scalar.h"

namespace scalar = hal::scalar;

extern "C" __attribute__((noinline, used)) void set_runtime_word(const std::uint32_t value)
{
    scalar::set(hal::gpr<5>(), value);
}

// CHECK-LABEL: <set_runtime_word>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_runtime_low(const std::uint32_t value)
{
    scalar::set_low(hal::gpr<5>(), value);
}

// CHECK-LABEL: <set_runtime_low>:
// CHECK-NEXT: lui [[LIMIT:a[0-7]]],0x10
// CHECK-NEXT: bgeu a0,[[LIMIT]],[[FAIL:[0-9a-f]+]]
// CHECK: [[FAIL]] <
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void set_runtime_high(const std::uint32_t value)
{
    scalar::set_high(hal::gpr<5>(), value);
}

// CHECK-LABEL: <set_runtime_high>:
// CHECK-NEXT: lui [[LIMIT:a[0-7]]],0x10
// CHECK-NEXT: bgeu a0,[[LIMIT]],[[FAIL:[0-9a-f]+]]
// CHECK: [[FAIL]] <
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_immediate(const std::uint32_t value)
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(value));
}

// CHECK-LABEL: <add_runtime_immediate>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: [[FAIL]] <
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_gpr(const std::uint32_t index)
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), hal::gpr(index));
}

// CHECK-LABEL: <add_runtime_gpr>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: [[FAIL]] <
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void compare_runtime_destination(const std::uint32_t index)
{
    scalar::compare<scalar::Compare::Equal>(hal::gpr(index), hal::gpr<2>(), scalar::immediate<3>());
}

// CHECK-LABEL: <compare_runtime_destination>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: [[FAIL]] <
// CHECK-NEXT: ebreak
