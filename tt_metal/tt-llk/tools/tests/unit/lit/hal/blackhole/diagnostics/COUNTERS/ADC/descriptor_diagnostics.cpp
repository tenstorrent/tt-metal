// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_unpack_thread}
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %t/constants.cpp -o %t/constants.o
// RUN: not %{diagnose} %t/no-clients.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{diagnose} %t/empty.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{diagnose} %t/empty-client-list.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY_CLIENTS
// RUN: not %{diagnose} %t/invalid-client.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CLIENT
// RUN: not %{diagnose} %t/increment-x.cpp 2>&1 | FileCheck %s --check-prefix=PACKED_RANGE
// RUN: not %{diagnose} %t/set-x.cpp 2>&1 | FileCheck %s --check-prefix=SINGLE_RANGE
// RUN: not %{diagnose} %t/set-y.cpp 2>&1 | FileCheck %s --check-prefix=SINGLE_RANGE
// RUN: not %{diagnose} %t/operation-two-instructions.cpp 2>&1 | FileCheck %s --check-prefix=TWO_INSTRUCTIONS
// RUN: not %{diagnose} %t/gpr-set.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// RUN: not %{diagnose} %t/gpr-operation.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// RUN: not %{diagnose} %t/gpr-runtime-index.cpp 2>&1 | FileCheck %s --check-prefix=REG2FLOP_DISABLED
// INVALID: error: static assertion failed: address-counter descriptor requires valid clients, an action, and at least one field
// EMPTY_CLIENTS: error: static assertion failed: select at least one address-counter client
// INVALID_CLIENT: error: static assertion failed: invalid address-counter client
// PACKED_RANGE: in 'constexpr' expansion of 'hal::detail::require({{.*}}"packed address-counter values must be in [0, 7]"))'
// PACKED_RANGE: error: '__builtin_trap()' is not a constant expression
// SINGLE_RANGE: in 'constexpr' expansion of 'hal::detail::require({{.*}}"SETADC value must be in [0, 65535] for X or [0, 8191] for Y/Z/W"))'
// SINGLE_RANGE: error: '__builtin_trap()' is not a constant expression
// TWO_INSTRUCTIONS: in 'constexpr' expansion of 'hal::detail::require({{.*}}"AddrCounter::operation() requires an update that encodes as one instruction; issue it with set()"))'
// TWO_INSTRUCTIONS: error: '__builtin_trap()' is not a constant expression
// REG2FLOP_DISABLED: error: static assertion failed: Blackhole does not handle REG2FLOP properly. GPR address-counter writes are unsupported
// clang-format on

//--- common.h
#pragma once

#include <cstdint>

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace adc      = hal::counters::adc;

//--- constants.cpp
#include "common.h"

constexpr auto counter = adc::AddrCounter {.channel0 = {.x = 1}}.for_clients<adc::Client::Unpacker0>();

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

//--- no-clients.cpp
#include "common.h"

void probe()
{
    counters::set<adc::AddrCounter {.channel0 = {.x = 1}}>();
}

//--- empty.cpp
#include "common.h"

void probe()
{
    counters::set<adc::AddrCounter {}.for_clients<adc::Client::Unpacker0>()>();
}

//--- empty-client-list.cpp
#include "common.h"

constexpr auto counter = adc::AddrCounter {.channel0 = {.x = 1}}.for_clients<>();

//--- invalid-client.cpp
#include "common.h"

constexpr auto counter = adc::AddrCounter {.channel0 = {.x = 1}}.for_clients<static_cast<adc::Client>(0x8)>();

//--- increment-x.cpp
#include "common.h"

void probe()
{
    counters::set<adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.x = 8}}.for_clients<adc::Client::Unpacker0>()>();
}

//--- set-x.cpp
#include "common.h"

void probe()
{
    counters::set<adc::AddrCounter {.channel0 = {.x = 0x10000}}.for_clients<adc::Client::Unpacker0>()>();
}

//--- set-y.cpp
#include "common.h"

void probe()
{
    counters::set<adc::AddrCounter {.channel0 = {.y = 0x2000}}.for_clients<adc::Client::Unpacker0>()>();
}

//--- operation-two-instructions.cpp
#include "common.h"

constexpr auto operation = adc::AddrCounter {.channel0 = {.x = 1, .z = 1}}.for_clients<adc::Client::Unpacker0>().operation();

//--- gpr-set.cpp
#include "common.h"

void probe()
{
    constexpr auto gpr_counter = adc::AddrCounter {.channel0 = {.x = hal::gpr<4>()}}.for_clients<adc::Client::Unpacker0>();
    counters::set(gpr_counter);
}

//--- gpr-operation.cpp
#include "common.h"

constexpr auto gpr_counter = adc::AddrCounter {.channel0 = {.w = hal::gpr<4>()}}.for_clients<adc::Client::Unpacker0>();
constexpr auto operation   = gpr_counter.operation();

//--- gpr-runtime-index.cpp
#include "common.h"

void probe(const std::uint32_t index)
{
    counters::set(adc::AddrCounter {.channel0 = {.x = hal::gpr(index)}}.for_clients<adc::Client::Unpacker0>());
}
