// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_math_thread}
// RUN: %split-file %s %t
// RUN: not %{diagnose} %t/run-raw-word.cpp 2>&1 | FileCheck %s --check-prefix=RUN_RAW
// RUN: not %{diagnose} %t/construct-from-word.cpp 2>&1 | FileCheck %s --check-prefix=CONSTRUCT
// RUN: not %{diagnose} %t/table-of-words.cpp 2>&1 | FileCheck %s --check-prefix=TABLE

// Only HAL encoders produce a hal::Operation, so a raw instruction word cannot be issued through hal::run.
// RUN_RAW: error: could not convert '{{.*}}' from 'unsigned int' to 'hal::Operation'
// CONSTRUCT: error: 'constexpr hal::Operation::Operation(uint32_t)' is private within this context
// TABLE: error: could not convert '{{.*}}' from 'unsigned int' to 'const hal::Operation'
// clang-format on

//--- run-raw-word.cpp
#include "hal/replay.h"

void test()
{
    hal::run(0x12345678u);
}

//--- construct-from-word.cpp
#include "hal/replay.h"

constexpr hal::Operation operation {0x12345678u};

//--- table-of-words.cpp
#include "hal/replay.h"

constexpr hal::Operation table[] = {hal::replay::get_operation(hal::replay::BufferRange {0, 4}), 0x12345678u};
