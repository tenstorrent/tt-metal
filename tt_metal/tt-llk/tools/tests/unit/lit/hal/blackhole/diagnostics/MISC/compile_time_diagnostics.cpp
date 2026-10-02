// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/flush-scope.cpp 2>&1 | FileCheck %s --check-prefix=FLUSH_SCOPE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-class.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_DESCRIPTOR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-resources.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_DESCRIPTOR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/run-linger.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_DESCRIPTOR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/constant-encoding.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_ENCODING
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// An invalid descriptor encoded in a constant expression is rejected by a trap.
// FLUSH_SCOPE: error: static assertion failed: TDMA flush selection is 4 bits
// INVALID_DESCRIPTOR: error: static assertion failed: invalid descriptor
// CONSTANT_ENCODING: error: '__builtin_trap()' is not a constant expression

//--- common.h
#pragma once

#include <cstdint>

#include "hal/misc.h"

namespace misc = hal::misc;

//--- flush-scope.cpp
#include "common.h"

void probe()
{
    misc::flush_tdma<static_cast<misc::FlushScope>(16)>();
}

//--- run-class.cpp
#include "common.h"

void probe()
{
    misc::run<misc::ResourceDeclaration {16, 0, 1}>();
}

//--- run-resources.cpp
#include "common.h"

void probe()
{
    misc::run<misc::ResourceDeclaration {0, 0x200, 1}>();
}

//--- run-linger.cpp
#include "common.h"

void probe()
{
    misc::run<misc::ResourceDeclaration {0, 0, 2048}>();
}

//--- constant-encoding.cpp
#include "common.h"

constexpr std::uint32_t operation = misc::ResourceDeclaration {16, 0, 1}.get_operation();
