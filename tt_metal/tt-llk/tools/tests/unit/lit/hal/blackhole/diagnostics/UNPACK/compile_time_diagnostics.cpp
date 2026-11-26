// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/valid-options.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/unset-handoff.cpp 2>&1 | FileCheck %s --check-prefix=UNSET_HANDOFF
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-increment.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_INCREMENT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-configuration-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CONFIGURATION_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-unpacker1-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_UNPACKER1_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-address-counter-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ADDRESS_COUNTER_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-transfer-issue.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_TRANSFER_ISSUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-increment-engine.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ENGINE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/unset-clear-handoff.cpp 2>&1 | FileCheck %s --check-prefix=UNSET_CLEAR_HANDOFF
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-clear-value.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CLEAR_VALUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/matrix-bank-handoff.cpp 2>&1 | FileCheck %s --check-prefix=MATRIX_BANK_HANDOFF
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/srcb-negative-infinity.cpp 2>&1 | FileCheck %s --check-prefix=SRCB_NEGATIVE_INFINITY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/both-banks-srcb-negative-infinity.cpp 2>&1 | FileCheck %s --check-prefix=BOTH_BANKS_SRCB_NEGATIVE_INFINITY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/invalid-nop-engine.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_NOP_ENGINE
// clang-format on

//--- valid-options.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

static_assert(unpack::is_valid(unpack::Transfer {.handoff = unpack::SrcHandoff::Keep}));
static_assert(unpack::is_valid(unpack::Transfer {
    .increments = {.channel0 = {.y = 3, .z = 3}, .channel1 = {.y = 3, .z = 3}},
    .context    = unpack::Context::explicit_context(7, 2),
    .handoff    = unpack::SrcHandoff::FlipAndSetDataValid,
}));
// Counter mode ignores the configuration ID, so unpacker 1 may still carry a large one.
static_assert(
    unpack::Unpacker1.operation().transfer<unpack::Transfer {.context = {unpack::ContextSource::Counter, 7, 0}, .handoff = unpack::SrcHandoff::Keep}>() != 0);
static_assert(!unpack::is_valid(unpack::Transfer {}));
static_assert(!unpack::is_valid(unpack::Transfer {
    .context = {static_cast<unpack::ContextSource>(3), 0, 0},
    .handoff = unpack::SrcHandoff::Keep,
}));
static_assert(unpack::is_valid(unpack::Clear {.value = unpack::ClearValue::One, .handoff = unpack::SrcHandoff::Keep, .wait = unpack::ClearWait::MatrixBank}));
static_assert(!unpack::is_valid(unpack::Clear {}));
static_assert(!unpack::is_valid(unpack::Clear {.value = static_cast<unpack::ClearValue>(3), .handoff = unpack::SrcHandoff::Keep}));
static_assert(!unpack::is_valid(unpack::Clear {.handoff = unpack::SrcHandoff::FlipAndSetDataValid, .wait = unpack::ClearWait::MatrixBank}));

// The runtime encoders are also usable in constant expressions.
static_assert(
    unpack::Unpacker1.operation().transfer({.handoff = unpack::SrcHandoff::Keep}) ==
    unpack::Unpacker1.operation().transfer<unpack::Transfer {.handoff = unpack::SrcHandoff::Keep}>());
static_assert(
    unpack::Unpacker0.operation().clear({.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::Keep}) ==
    unpack::Unpacker0.operation().clear<unpack::Clear {.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::Keep}>());
static_assert(
    unpack::Unpacker0.operation().flush_row_start_cache(unpack::CacheScope::AllEntries) ==
    unpack::Unpacker0.operation().flush_row_start_cache<unpack::CacheScope::AllEntries>());
static_assert(
    unpack::experimental::clear_both_banks_operation(unpack::Unpacker1, {}) ==
    unpack::experimental::clear_both_banks_operation<unpack::experimental::BothBanksClear {}>(unpack::Unpacker1));

// An unpacker built from a constant engine encodes like the predefined one.
static_assert(
    unpack::Unpacker {unpack::Engine::Unpacker1}.operation().increment_context_counter() == unpack::Unpacker1.operation().increment_context_counter());

//--- unset-handoff.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker0.operation().transfer<unpack::Transfer {}>();
// UNSET_HANDOFF: error: static assertion failed: invalid UNPACR data-transfer options

//--- invalid-increment.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker0.operation()
                               .transfer<unpack::Transfer {
                                   .increments = {.channel1 = {.z = 4}},
                                   .handoff    = unpack::SrcHandoff::Keep,
                               }>();
// INVALID_INCREMENT: error: static assertion failed: invalid UNPACR data-transfer options

//--- invalid-configuration-context.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker0.operation()
                               .transfer<unpack::Transfer {
                                   .context = unpack::Context::explicit_context(8),
                                   .handoff = unpack::SrcHandoff::Keep,
                               }>();
// INVALID_CONFIGURATION_CONTEXT: error: static assertion failed: invalid UNPACR data-transfer options

//--- invalid-unpacker1-context.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker1.operation()
                               .transfer<unpack::Transfer {
                                   .context = unpack::Context::explicit_context(2),
                                   .handoff = unpack::SrcHandoff::Keep,
                               }>();
// INVALID_UNPACKER1_CONTEXT: in 'constexpr' expansion of 'hal::detail::require({{.*}}"UNPACR unpacker-1 explicit configuration context must be in [0, 1]"))'
// INVALID_UNPACKER1_CONTEXT: error: '__builtin_trap()' is not a constant expression

//--- invalid-address-counter-context.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker0.operation()
                               .transfer<unpack::Transfer {
                                   .context = unpack::Context::counter(3),
                                   .handoff = unpack::SrcHandoff::Keep,
                               }>();
// INVALID_ADDRESS_COUNTER_CONTEXT: error: static assertion failed: invalid UNPACR data-transfer options

//--- invalid-transfer-issue.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

void issue()
{
    unpack::Unpacker1.transfer<unpack::Transfer {}>();
}

// INVALID_TRANSFER_ISSUE: error: static assertion failed: invalid UNPACR data-transfer options

//--- invalid-increment-engine.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker {static_cast<unpack::Engine>(2)}.operation().increment_context_counter();
// INVALID_ENGINE: in 'constexpr' expansion of 'hal::detail::require({{.*}}"UNPACR engine must be Unpacker0 or Unpacker1"))'
// INVALID_ENGINE: error: '__builtin_trap()' is not a constant expression

//--- unset-clear-handoff.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

void issue()
{
    unpack::Unpacker0.clear<unpack::Clear {.value = unpack::ClearValue::NegativeInfinity}>();
}

// UNSET_CLEAR_HANDOFF: error: static assertion failed: invalid UNPACR_NOP source-clear options

//--- invalid-clear-value.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation =
    unpack::Unpacker0.operation().clear<unpack::Clear {.value = static_cast<unpack::ClearValue>(3), .handoff = unpack::SrcHandoff::Keep}>();
// INVALID_CLEAR_VALUE: error: static assertion failed: invalid UNPACR_NOP source-clear options

//--- matrix-bank-handoff.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation = unpack::Unpacker0.operation().clear({.handoff = unpack::SrcHandoff::FlipAndSetDataValid, .wait = unpack::ClearWait::MatrixBank});
// MATRIX_BANK_HANDOFF: in 'constexpr' expansion of 'hal::detail::require({{.*}}"UNPACR_NOP FlipAndSetDataValid requires ClearWait::UnpackerBank"))'
// MATRIX_BANK_HANDOFF: error: '__builtin_trap()' is not a constant expression

//--- srcb-negative-infinity.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation =
    unpack::Unpacker1.operation().clear<unpack::Clear {.value = unpack::ClearValue::NegativeInfinity, .handoff = unpack::SrcHandoff::Keep}>();
// SRCB_NEGATIVE_INFINITY: in 'constexpr' expansion of 'hal::detail::require({{.*}}"UNPACR_NOP cannot clear SrcB to negative infinity"))'
// SRCB_NEGATIVE_INFINITY: error: '__builtin_trap()' is not a constant expression

//--- both-banks-srcb-negative-infinity.cpp
#include "hal/unpack.h"

namespace unpack = hal::unpack;

constexpr auto operation =
    unpack::experimental::clear_both_banks_operation<unpack::experimental::BothBanksClear {.value = unpack::ClearValue::NegativeInfinity}>(unpack::Unpacker1);
// BOTH_BANKS_SRCB_NEGATIVE_INFINITY: in 'constexpr' expansion of 'hal::detail::require({{.*}}"UNPACR_NOP cannot clear SrcB to negative infinity"))'
// BOTH_BANKS_SRCB_NEGATIVE_INFINITY: error: '__builtin_trap()' is not a constant expression

//--- invalid-nop-engine.cpp
#include "hal/nop.h"

constexpr auto operation = hal::nop::unpacker_operation<static_cast<hal::unpack::Engine>(2)>();
// INVALID_NOP_ENGINE: error: static assertion failed: UNPACR_NOP engine must be Unpacker0 or Unpacker1
