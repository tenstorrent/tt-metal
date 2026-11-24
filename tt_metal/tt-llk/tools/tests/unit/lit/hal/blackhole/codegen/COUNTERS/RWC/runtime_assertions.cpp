// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// Each runtime check is called with the last valid and the first invalid value, which the compiler folds.
// The test checks only whether the LLK_ASSERT ebreak is emitted, so it does not depend on how the comparison is compiled.

#include <cstdint>

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace rwc      = hal::counters::rwc;

extern "C" __attribute__((noinline, used)) void set_value_valid()
{
    const auto counter = rwc::MathCounter {.dest = rwc::Set(std::uint32_t {15})};
    counters::set(counter);
}

// CHECK-LABEL: <set_value_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_value_invalid()
{
    const auto counter = rwc::MathCounter {.dest = rwc::Set(std::uint32_t {16})};
    counters::set(counter);
}

// CHECK-LABEL: <set_value_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void increment_value_valid()
{
    const auto counter = rwc::MathCounter {.src_a = rwc::Increment(std::uint32_t {15})};
    counters::set(counter);
}

// CHECK-LABEL: <increment_value_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void increment_value_invalid()
{
    const auto counter = rwc::MathCounter {.src_a = rwc::Increment(std::uint32_t {16})};
    counters::set(counter);
}

// CHECK-LABEL: <increment_value_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void update_something_valid()
{
    const auto counter = rwc::MathCounter {.clear_fidelity = true};
    counters::set(counter);
}

// CHECK-LABEL: <update_something_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void update_something_invalid()
{
    const auto counter = rwc::MathCounter {};
    counters::set(counter);
}

// CHECK-LABEL: <update_something_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void increment_and_save_on_dest_valid()
{
    const rwc::DestUpdate save = rwc::IncrementAndSave(std::uint32_t {2});
    const auto counter         = rwc::MathCounter {.dest = save};
    counters::set(counter);
}

// CHECK-LABEL: <increment_and_save_on_dest_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void increment_and_save_on_dest_invalid()
{
    const rwc::DestUpdate save = rwc::IncrementAndSave(std::uint32_t {2});
    const auto counter         = rwc::MathCounter {.src_a = save};
    counters::set(counter);
}

// CHECK-LABEL: <increment_and_save_on_dest_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t operation_one_instruction_valid()
{
    const auto counter = rwc::MathCounter {.src_a = rwc::Set(std::uint32_t {1}), .dest = rwc::Set(std::uint32_t {2})};
    return counter.operation();
}

// CHECK-LABEL: <operation_one_instruction_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t operation_one_instruction_invalid()
{
    const auto counter = rwc::MathCounter {.src_a = rwc::Set(std::uint32_t {1}), .dest = rwc::Increment(std::uint32_t {2})};
    return counter.operation();
}

// CHECK-LABEL: <operation_one_instruction_invalid>:
// CHECK: ebreak
