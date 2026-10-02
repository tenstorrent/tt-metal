// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/all-index.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ZERO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/half-index.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ZERO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/face-index.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ZERO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/row-index.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ZERO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/address-mode.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ZERO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/scope.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ZERO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/constant-encoding.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_ENCODING
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// INVALID_ZERO: error: static assertion failed: invalid ZEROACC descriptor
// Constant evaluation of get_operation() rejects an invalid descriptor by trapping.
// CONSTANT_ENCODING: error: '__builtin_trap()' is not a constant expression

//--- all-index.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

void f()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::All, .index = 1}>();
}

//--- half-index.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

void f()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .index = 2}>();
}

//--- face-index.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

void f()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::Face, .index = 256}>();
}

//--- row-index.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

void f()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 1u << 14}>();
}

//--- address-mode.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

void f()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::SingleRow, .address_mode = 8}>();
}

//--- scope.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

void f()
{
    dst::run<dst::Zero {.scope = static_cast<dst::ZeroScope>(4)}>();
}

//--- constant-encoding.cpp
#include "hal/dst.h"

namespace dst = hal::dst;

constexpr auto operation = dst::Zero {.scope = dst::ZeroScope::All, .index = 1}.get_operation();
