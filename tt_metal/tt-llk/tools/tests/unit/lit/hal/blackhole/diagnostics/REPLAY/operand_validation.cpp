// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -Werror=narrowing %t/operation-int-count.cpp 2>&1 | FileCheck %s --check-prefix=INT_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-runtime-template-count.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_TEMPLATE
// clang-format on

// BufferRange fields are std::uint32_t; tt-metal and LLK builds turn the narrowing
// warning into an error with -Werror.
// INT_COUNT: error: narrowing conversion of '(((bool)need_stall) ? 3 : 2)' from 'int' to 'uint32_t'

// A value computed from function arguments needs the runtime overload.
// RUNTIME_TEMPLATE: error: the value of 'count' is not usable in a constant expression

//--- operation-int-count.cpp
#include <cstdint>

#include "hal/replay.h"

std::uint32_t probe(const bool need_stall)
{
    return hal::replay::get_operation({22, need_stall ? 3 : 2});
}

//--- run-runtime-template-count.cpp
#include <cstdint>

#include "hal/replay.h"

void probe(const std::uint32_t face_r_dim)
{
    const std::uint32_t count = face_r_dim - 1;
    hal::replay::run<hal::replay::BufferRange {0, count}>();
}
