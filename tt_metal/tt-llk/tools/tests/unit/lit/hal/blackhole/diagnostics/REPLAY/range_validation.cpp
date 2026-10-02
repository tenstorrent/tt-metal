// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/record-start-outside.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/record-count-zero.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-start-outside.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-count-zero.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-start-outside.cpp 2>&1 | FileCheck %s
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-count-too-large.cpp 2>&1 | FileCheck %s
// clang-format on

// CHECK: error: static assertion failed: Replay range requires start < 32 and count in [1, 64]

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
    return hal::replay::get_operation<hal::replay::BufferRange {0, 65}>();
}
