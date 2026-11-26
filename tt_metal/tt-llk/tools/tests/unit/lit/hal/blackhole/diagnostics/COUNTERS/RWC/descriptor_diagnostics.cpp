// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_math_thread}
// RUN: %split-file %s %t
// RUN: not %{diagnose} %t/empty.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: not %{diagnose} %t/src-increment-and-save.cpp 2>&1 | FileCheck %s --check-prefix=DEST_ONLY
// RUN: not %{diagnose} %t/sliced-increment-and-save.cpp 2>&1 | FileCheck %s --check-prefix=SLICED_SAVE
// RUN: not %{diagnose} %t/set-immediate.cpp 2>&1 | FileCheck %s --check-prefix=VALUE_RANGE
// RUN: not %{diagnose} %t/increment-immediate.cpp 2>&1 | FileCheck %s --check-prefix=VALUE_RANGE
// RUN: not %{diagnose} %t/operation-two-instructions.cpp 2>&1 | FileCheck %s --check-prefix=TWO_INSTRUCTIONS
// EMPTY: error: static assertion failed: math-counter descriptor must update a counter, flip a bank, or clear the fidelity phase
// DEST_ONLY: error: could not convert {{.*}}IncrementAndSave{{.*}} to 'hal::counters::rwc::SrcUpdate'
// SLICED_SAVE: in 'constexpr' expansion of 'hal::detail::require({{.*}}"IncrementAndSave applies only to the Dest counter"))'
// SLICED_SAVE: error: '__builtin_trap()' is not a constant expression
// VALUE_RANGE: in 'constexpr' expansion of 'hal::detail::require({{.*}}"math-counter value must fit in four bits"))'
// VALUE_RANGE: error: '__builtin_trap()' is not a constant expression
// TWO_INSTRUCTIONS: in 'constexpr' expansion of 'hal::detail::require({{.*}}"MathCounter::operation() encodes one instruction, but this descriptor needs SETRWC and INCRWC{{.*}}"))'
// clang-format on

//--- common.h
#pragma once

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace rwc      = hal::counters::rwc;

//--- empty.cpp
#include "common.h"

void probe()
{
    counters::set<rwc::MathCounter {}>();
}

//--- src-increment-and-save.cpp
#include "common.h"

constexpr auto save = rwc::MathCounter {.src_a = rwc::IncrementAndSave(1)};

//--- sliced-increment-and-save.cpp
#include "common.h"

// A DestUpdate slices into SrcUpdate, so the conversion error in src-increment-and-save.cpp does not apply.
constexpr rwc::DestUpdate save = rwc::IncrementAndSave(1);
constexpr auto operation       = rwc::MathCounter {.src_a = save}.operation();

//--- set-immediate.cpp
#include "common.h"

void probe()
{
    counters::set<rwc::MathCounter {.dest = rwc::Set(16)}>();
}

//--- increment-immediate.cpp
#include "common.h"

constexpr auto operation = rwc::MathCounter {.src_a = rwc::Increment(16)}.operation();

//--- operation-two-instructions.cpp
#include "common.h"

constexpr auto operation = rwc::MathCounter {.src_a = rwc::Set(1), .dest = rwc::Increment(2)}.operation();
