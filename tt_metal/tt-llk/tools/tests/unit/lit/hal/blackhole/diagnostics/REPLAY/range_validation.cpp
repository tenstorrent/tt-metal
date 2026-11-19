// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/record-start-outside.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/record-count-zero.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-start-outside.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-count-zero.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-past-last-slot.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-start-outside.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-count-too-large.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-constant-past-last-slot.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_EVALUATION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-negative-count.cpp 2>&1 | FileCheck %s --check-prefix=NEGATIVE_COUNT
// clang-format on

// CHECK: error: static assertion failed: Replay range requires count >= 1 and start + count <= 32
// CONSTANT_EVALUATION: in 'constexpr' expansion of 'hal::detail::require({{.*}}"Replay range requires count >= 1 and start + count <= 32"))'
// CONSTANT_EVALUATION: error: '__builtin_trap()' is not a constant expression
// NEGATIVE_COUNT: error: narrowing conversion of '-1' from 'int' to 'uint32_t'

//--- record-start-outside.cpp
#include "hal/replay.h"

void probe()
{
    hal::replay::record<hal::replay::BufferRange {32, 1}>([] { TTI_NOP; });
}

//--- record-count-zero.cpp
#include "hal/replay.h"

void probe()
{
    hal::replay::record<hal::replay::BufferRange {0, 0}>([] {});
}

//--- run-start-outside.cpp
#include "hal/replay.h"

void probe()
{
    hal::replay::run<hal::replay::BufferRange {32, 1}>();
}

//--- run-count-zero.cpp
#include "hal/replay.h"

void probe()
{
    hal::replay::run<hal::replay::BufferRange {0, 0}>();
}

//--- run-past-last-slot.cpp
#include "hal/replay.h"

void probe()
{
    hal::replay::run<hal::replay::BufferRange {31, 2}>();
}

//--- operation-start-outside.cpp
#include <cstdint>

#include "hal/replay.h"

std::uint32_t probe()
{
    return hal::replay::get_operation<hal::replay::BufferRange {32, 1}>();
}

//--- operation-count-too-large.cpp
#include <cstdint>

#include "hal/replay.h"

std::uint32_t probe()
{
    return hal::replay::get_operation<hal::replay::BufferRange {0, 33}>();
}

//--- operation-constant-past-last-slot.cpp
#include <cstdint>

#include "hal/replay.h"

// The runtime overload rejects an invalid range during constant evaluation.
constexpr std::uint32_t fragment(const std::uint32_t start, const std::uint32_t count)
{
    return hal::replay::get_operation({start, count});
}

constexpr std::uint32_t operation = fragment(30, 4);

//--- run-negative-count.cpp
#include "hal/replay.h"

// A count derived from a constant expression that goes negative narrows before the
// range check runs.
template <int Iterations>
void probe()
{
    hal::replay::run<hal::replay::BufferRange {0, 7 * (Iterations % 4) - 1}>();
}

template void probe<28>();
