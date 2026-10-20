// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_unpack_thread}
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %t/constants.cpp -o %t/constants.o
// RUN: not %{diagnose} %t/apply.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// RUN: not %{diagnose} %t/get-operation.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// RUN: not %{diagnose} %t/current-thread.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// RUN: not %{diagnose} %t/carry-halfword.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// RUN: not %{diagnose} %t/encoded-byte.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// REG2FLOP_DISABLED: error: static assertion failed: Blackhole does not handle REG2FLOP properly. GPR address-counter writes are unsupported
// clang-format on

//--- common.h
#pragma once

#include "hal/counters.h"

namespace counters = hal::counters;
namespace adc      = hal::counters::adc;

constexpr auto counter = adc::AddrCounter {.channel0 = {.x = 1}}.for_clients<adc::Client::Unpacker0>();

//--- constants.cpp
#include "common.h"

static_assert(counter.operation() != 0);

void probe()
{
    constexpr auto counter_increment = adc::AddrCounter {.action = adc::Action::Increment, .channel0 = counter.channel0}.for_clients<adc::Client::Unpacker0>();
    constexpr auto counter_reload =
        adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = counter.channel0}.for_clients<adc::Client::Unpacker0>();

    counters::set<counter>();
    counters::set<counter_increment>();
    counters::set<counter_reload>();
}

//--- apply.cpp
#include "common.h"

void probe()
{
    constexpr auto gpr_counter = adc::AddrCounter {.channel0 = {.x = hal::gpr<4>()}}.for_clients<adc::Client::Unpacker0>();
    counters::set(gpr_counter);
}

//--- get-operation.cpp
#include "common.h"

constexpr auto gpr_counter = adc::AddrCounter {.channel0 = {.w = hal::gpr<4>()}}.for_clients<adc::Client::Unpacker0>();
constexpr auto operation   = gpr_counter.operation();

// The old codegen cases expected REG2FLOP instructions. Every GPR source is now rejected,
// before selecting a thread, a carry destination, or a source-byte slice.

//--- current-thread.cpp
#include "common.h"

void reg2flop_current_thread_counter()
{
    constexpr auto gpr_counter = adc::AddrCounter {.channel0 = {.x = hal::gpr<5>()}}.for_clients<adc::Client::Unpacker0>();
    counters::set(gpr_counter);
}

//--- carry-halfword.cpp
#include "common.h"

void reg2flop_other_thread_carry_halfword()
{
    constexpr auto gpr_counter = adc::AddrCounter {.channel1 = {.w = hal::gpr<60>()}}.for_clients<adc::Client::Packers>();
    counters::set(gpr_counter);
}

//--- encoded-byte.cpp
#include "common.h"

constexpr auto gpr_counter                          = adc::AddrCounter {.channel0 = {.y = hal::gpr<17>()}}.for_clients<adc::Client::Unpacker1>();
constexpr auto reg2flop_byte_from_encoded_operation = gpr_counter.operation();
