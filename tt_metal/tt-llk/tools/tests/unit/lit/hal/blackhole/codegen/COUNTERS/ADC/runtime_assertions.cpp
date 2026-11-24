// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// Each runtime check is called with the last valid and the first invalid value, which the compiler folds.
// The test checks only whether the LLK_ASSERT ebreak is emitted, so it does not depend on how the comparison is compiled.

#include <cstdint>

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace adc      = hal::counters::adc;

extern "C" __attribute__((noinline, used)) void set_x_valid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0xffff}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_x_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_x_invalid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0x10000}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_x_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void set_w_valid()
{
    const auto counter = adc::AddrCounter {.channel1 = {.w = std::uint32_t {0x1fff}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_w_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_w_invalid()
{
    const auto counter = adc::AddrCounter {.channel1 = {.w = std::uint32_t {0x2000}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_w_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void set_dual_x_valid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0}}, .channel1 = {.x = std::uint32_t {1023}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_dual_x_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_dual_x_invalid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0}}, .channel1 = {.x = std::uint32_t {1024}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_dual_x_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void set_packed_valid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0}, .y = std::uint32_t {7}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_packed_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void set_packed_invalid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0}, .y = std::uint32_t {8}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <set_packed_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void increment_valid()
{
    const auto counter = adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.z = std::uint32_t {7}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <increment_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void increment_invalid()
{
    const auto counter = adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.z = std::uint32_t {8}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <increment_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void select_counter_valid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {0}}}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <select_counter_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void select_counter_invalid()
{
    const auto counter = adc::AddrCounter {}.for_clients<adc::Client::Unpacker0>();
    counters::set(counter);
}

// CHECK-LABEL: <select_counter_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t operation_one_instruction_valid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {1}}}.for_clients<adc::Client::Unpacker0>();
    return counter.operation();
}

// CHECK-LABEL: <operation_one_instruction_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t operation_one_instruction_invalid()
{
    const auto counter = adc::AddrCounter {.channel0 = {.x = std::uint32_t {1}, .z = std::uint32_t {1}}}.for_clients<adc::Client::Unpacker0>();
    return counter.operation();
}

// CHECK-LABEL: <operation_one_instruction_invalid>:
// CHECK: ebreak
