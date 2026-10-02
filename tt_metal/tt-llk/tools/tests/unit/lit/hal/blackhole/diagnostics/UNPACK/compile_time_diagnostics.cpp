// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/valid-descriptors.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/unset-handoff.cpp 2>&1 | FileCheck %s --check-prefix=UNSET_HANDOFF
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-increment.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_INCREMENT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-configuration-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CONFIGURATION_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-unpacker1-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_UNPACKER1_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-address-counter-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ADDRESS_COUNTER_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-transfer-issue.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_TRANSFER_ISSUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-increment-engine.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_INCREMENT_ENGINE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-flush-engine.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_FLUSH_ENGINE
// clang-format on

//--- valid-descriptors.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr unpack::Engine invalid_engine = static_cast<unpack::Engine>(2);

static_assert(unpack::is_valid(unpack::DataTransfer {.engine = unpack::Engine::Unpacker0, .handoff = unpack::SourceHandoff::Keep}));
static_assert(unpack::is_valid(unpack::DataTransfer {
    .engine     = unpack::Engine::Unpacker1,
    .increments = {.channel0 = {.y = 3, .z = 3}, .channel1 = {.y = 3, .z = 3}},
    .context    = unpack::ContextSelection::explicit_context(1, 2),
    .handoff    = unpack::SourceHandoff::FlipAndSetDataValid,
}));
// Counter mode ignores the configuration ID, so unpacker 1 may still carry a large one.
static_assert(unpack::is_valid(unpack::DataTransfer {
    .engine  = unpack::Engine::Unpacker1,
    .context = {unpack::ContextSource::Counter, 7, 0},
    .handoff = unpack::SourceHandoff::Keep,
}));
static_assert(!unpack::is_valid(unpack::DataTransfer {.engine = unpack::Engine::Unpacker0}));
static_assert(!unpack::is_valid(unpack::DataTransfer {.engine = invalid_engine, .handoff = unpack::SourceHandoff::Keep}));
static_assert(!unpack::is_valid(unpack::DataTransfer {
    .engine  = unpack::Engine::Unpacker0,
    .context = {static_cast<unpack::ContextSource>(3), 0, 0},
    .handoff = unpack::SourceHandoff::Keep,
}));
static_assert(unpack::is_valid(unpack::ContextCounterIncrement {unpack::Engine::Unpacker1}));
static_assert(!unpack::is_valid(unpack::ContextCounterIncrement {invalid_engine}));
static_assert(unpack::is_valid(unpack::RowStartCacheFlush {unpack::Engine::Unpacker1, unpack::CacheScope::AllEntries}));
static_assert(!unpack::is_valid(unpack::RowStartCacheFlush {invalid_engine}));

// The runtime encoder is also usable in constant expressions.
static_assert(
    unpack::get_operation(unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::Keep}) ==
    unpack::get_operation<unpack::DataTransfer {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::Keep}>());
static_assert(
    unpack::get_operation(unpack::ContextCounterIncrement {unpack::Engine::Unpacker0}) ==
    unpack::get_operation<unpack::ContextCounterIncrement {unpack::Engine::Unpacker0}>());
static_assert(
    unpack::get_operation(unpack::RowStartCacheFlush {unpack::Engine::Unpacker0}) ==
    unpack::get_operation<unpack::RowStartCacheFlush {unpack::Engine::Unpacker0}>());

//--- unset-handoff.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::get_operation<unpack::DataTransfer {.engine = unpack::Engine::Unpacker0}>();
// UNSET_HANDOFF: error: static assertion failed: invalid UNPACR data-transfer descriptor

//--- invalid-increment.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::get_operation<unpack::DataTransfer {
    .engine     = unpack::Engine::Unpacker0,
    .increments = {.channel1 = {.z = 4}},
    .handoff    = unpack::SourceHandoff::Keep,
}>();
// INVALID_INCREMENT: error: static assertion failed: invalid UNPACR data-transfer descriptor

//--- invalid-configuration-context.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::get_operation<unpack::DataTransfer {
    .engine  = unpack::Engine::Unpacker0,
    .context = unpack::ContextSelection::explicit_context(8),
    .handoff = unpack::SourceHandoff::Keep,
}>();
// INVALID_CONFIGURATION_CONTEXT: error: static assertion failed: invalid UNPACR data-transfer descriptor

//--- invalid-unpacker1-context.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::get_operation<unpack::DataTransfer {
    .engine  = unpack::Engine::Unpacker1,
    .context = unpack::ContextSelection::explicit_context(2),
    .handoff = unpack::SourceHandoff::Keep,
}>();
// INVALID_UNPACKER1_CONTEXT: error: static assertion failed: invalid UNPACR data-transfer descriptor

//--- invalid-address-counter-context.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::get_operation<unpack::DataTransfer {
    .engine  = unpack::Engine::Unpacker0,
    .context = unpack::ContextSelection::counter(3),
    .handoff = unpack::SourceHandoff::Keep,
}>();
// INVALID_ADDRESS_COUNTER_CONTEXT: error: static assertion failed: invalid UNPACR data-transfer descriptor

//--- invalid-transfer-issue.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

void issue()
{
    unpack::run<unpack::DataTransfer {.engine = unpack::Engine::Unpacker1}>();
}

// INVALID_TRANSFER_ISSUE: error: static assertion failed: invalid UNPACR data-transfer descriptor

//--- invalid-increment-engine.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

void issue()
{
    unpack::run<unpack::ContextCounterIncrement {static_cast<unpack::Engine>(2)}>();
}

// INVALID_INCREMENT_ENGINE: error: static assertion failed: invalid UNPACR context-counter increment descriptor

//--- invalid-flush-engine.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

void issue()
{
    unpack::run<unpack::RowStartCacheFlush {static_cast<unpack::Engine>(2), unpack::CacheScope::AllEntries}>();
}

// INVALID_FLUSH_ENGINE: error: static assertion failed: invalid UNPACR row-start-cache flush descriptor
