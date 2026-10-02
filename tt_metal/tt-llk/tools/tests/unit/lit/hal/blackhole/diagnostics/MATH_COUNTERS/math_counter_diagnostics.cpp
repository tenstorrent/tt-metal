// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/no-operation.cpp 2>&1 | FileCheck %s --check-prefix=NO_OPERATION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mixed-modes.cpp 2>&1 | FileCheck %s --check-prefix=MIXED_MODES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/empty-set.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY_SET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/reassigned.cpp 2>&1 | FileCheck %s --check-prefix=REASSIGNED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/set-immediate.cpp 2>&1 | FileCheck %s --check-prefix=SET_IMMEDIATE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/increment-immediate.cpp 2>&1 | FileCheck %s --check-prefix=INCREMENT_IMMEDIATE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/bias-immediate.cpp 2>&1 | FileCheck %s --check-prefix=BIAS_IMMEDIATE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/carry-without-entry.cpp 2>&1 | FileCheck %s --check-prefix=CARRY_WITHOUT_ENTRY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/release-dest.cpp 2>&1 | FileCheck %s --check-prefix=RELEASE_DEST
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/increment-release.cpp 2>&1 | FileCheck %s --check-prefix=INCREMENT_RELEASE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/increment-clear-fidelity.cpp 2>&1 | FileCheck %s --check-prefix=INCREMENT_CLEAR_FIDELITY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/dest-save-and-carry.cpp 2>&1 | FileCheck %s --check-prefix=DEST_SAVE_AND_CARRY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-two-instructions.cpp 2>&1 | FileCheck %s --check-prefix=OPERATION_TWO_INSTRUCTIONS
// clang-format on

//--- no-operation.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

void no_operation()
{
    hal::math_counters.apply();
}

// NO_OPERATION: error: static assertion failed: no math-counter operation selected — call set<>(), increment<>(), release<>(), or clear_fidelity() first

//--- mixed-modes.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto mixed_modes = hal::math_counters.set<Counters::SrcA, 1>().increment<Counters::SrcB, 1>();
// MIXED_MODES: error: static assertion failed: cannot mix set and increment entries

//--- empty-set.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto empty_set = hal::math_counters.set<Counters::None, 1>();
// EMPTY_SET: error: static assertion failed: set entry must select at least one counter

//--- reassigned.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto reassigned = hal::math_counters.set<Counters::SrcA | Counters::Dest, 1>().set<Counters::Dest, 2>();
// REASSIGNED: error: static assertion failed: counter already assigned by an earlier entry

//--- set-immediate.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto set_immediate = hal::math_counters.set<Counters::Dest, 16>();
// SET_IMMEDIATE: error: static assertion failed: SrcA/SrcB/Dest SETRWC immediates must fit four bits

//--- increment-immediate.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto increment_immediate = hal::math_counters.increment<Counters::SrcA, 16>();
// INCREMENT_IMMEDIATE: error: static assertion failed: SrcA/SrcB/Dest INCRWC immediates must fit four bits

//--- bias-immediate.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto bias_immediate = hal::math_counters.set<Counters::Bias, 4096>();
// BIAS_IMMEDIATE: error: static assertion failed: Bias SETIBRWC immediate must fit twelve bits

//--- carry-without-entry.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

void carry_without_entry()
{
    hal::math_counters.set<Counters::SrcA, 1>().advance_carry_and_reload<Counters::SrcB>().apply();
}

// CARRY_WITHOUT_ENTRY: error: static assertion failed: carry advance requires a matching set/increment entry

//--- release-dest.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto release_dest = hal::math_counters.release<Counters::Dest>();
// RELEASE_DEST: error: static assertion failed: release can select only SrcA and SrcB

//--- increment-release.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto increment_release = hal::math_counters.increment<Counters::SrcA, 1>().release<Counters::SrcA>();
// INCREMENT_RELEASE: error: static assertion failed: INCRWC cannot release source banks

//--- increment-clear-fidelity.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto increment_clear_fidelity = hal::math_counters.increment<Counters::SrcA, 1>().clear_fidelity();
// INCREMENT_CLEAR_FIDELITY: error: static assertion failed: INCRWC cannot clear the fidelity phase

//--- dest-save-and-carry.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

void dest_save_and_carry()
{
    hal::math_counters.advance_dest_and_save_to_carry<1>().advance_carry_and_reload<Counters::Dest>().apply();
}

// DEST_SAVE_AND_CARRY: error: static assertion failed: Dest cannot advance from carry and save to carry together

//--- operation-two-instructions.cpp
#include "hal/math_counters.h"

using hal::rwc::Counters;

constexpr auto two_instructions = hal::math_counters.set<Counters::SrcA | Counters::Bias, 4>().get_operation();
// OPERATION_TWO_INSTRUCTIONS: error: static assertion failed: get_operation() requires exactly one instruction; use apply() when Bias is combined with
// SETRWC/INCRWC state
