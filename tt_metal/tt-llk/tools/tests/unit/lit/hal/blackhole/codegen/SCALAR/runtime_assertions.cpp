// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// Each runtime check is called with the last valid and the first invalid value, which the compiler folds.
// The test checks only whether the LLK_ASSERT ebreak is emitted, so it does not depend on how the comparison is compiled.

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

extern "C" __attribute__((noinline, used)) void set_runtime_low_valid()
{
    scalar::set_low(hal::gpr<5>(), std::uint32_t {0xffff});
}

// CHECK-LABEL: <set_runtime_low_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_runtime_low_invalid()
{
    scalar::set_low(hal::gpr<5>(), std::uint32_t {0x10000});
}

// CHECK-LABEL: <set_runtime_low_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void set_runtime_high_valid()
{
    scalar::set_high(hal::gpr<5>(), std::uint32_t {0xffff});
}

// CHECK-LABEL: <set_runtime_high_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_runtime_high_invalid()
{
    scalar::set_high(hal::gpr<5>(), std::uint32_t {0x10000});
}

// CHECK-LABEL: <set_runtime_high_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_immediate_valid()
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(std::uint32_t {63}));
}

// CHECK-LABEL: <add_runtime_immediate_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_immediate_invalid()
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(std::uint32_t {64}));
}

// CHECK-LABEL: <add_runtime_immediate_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void shift_runtime_immediate_valid()
{
    scalar::shift_left(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(std::uint32_t {31}));
}

// CHECK-LABEL: <shift_runtime_immediate_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void shift_runtime_immediate_invalid()
{
    scalar::shift_left(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(std::uint32_t {32}));
}

// CHECK-LABEL: <shift_runtime_immediate_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_gpr_valid()
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), hal::gpr(std::uint32_t {63}));
}

// CHECK-LABEL: <add_runtime_gpr_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void add_runtime_gpr_invalid()
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), hal::gpr(std::uint32_t {64}));
}

// CHECK-LABEL: <add_runtime_gpr_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void compare_runtime_destination_valid()
{
    scalar::compare<scalar::Compare::Equal>(hal::gpr(std::uint32_t {63}), hal::gpr<2>(), scalar::immediate<3>());
}

// CHECK-LABEL: <compare_runtime_destination_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void compare_runtime_destination_invalid()
{
    scalar::compare<scalar::Compare::Equal>(hal::gpr(std::uint32_t {64}), hal::gpr<2>(), scalar::immediate<3>());
}

// CHECK-LABEL: <compare_runtime_destination_invalid>:
// CHECK: ebreak
