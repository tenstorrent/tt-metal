// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/eltwise-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=ELTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/eltwise-operation.cpp 2>&1 | FileCheck %s --check-prefix=ELTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/eltwise-broadcast.cpp 2>&1 | FileCheck %s --check-prefix=ELTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/eltwise-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=ELTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/eltwise-dest-row-offset.cpp 2>&1 | FileCheck %s --check-prefix=ELTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matmul-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=MATMUL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matmul-broadcast.cpp 2>&1 | FileCheck %s --check-prefix=MATMUL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matmul-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=MATMUL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matmul-row-broadcast-offset.cpp 2>&1 | FileCheck %s --check-prefix=MATMUL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-row-broadcast-offset.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_ROW_BROADCAST
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/pool-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=POOL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/pool-sum-indices.cpp 2>&1 | FileCheck %s --check-prefix=POOL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/pool-function.cpp 2>&1 | FileCheck %s --check-prefix=POOL
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// ELTWISE: error: static assertion failed: invalid eltwise options
// MATMUL: error: static assertion failed: invalid matmul options
// POOL: error: static assertion failed: invalid pool options

//--- eltwise-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add}>();
}

//--- eltwise-operation.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = static_cast<fpu::EltwiseOperation>(3), .release = fpu::SourceRelease::None}>();
}

//--- eltwise-broadcast.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {
        .operation = fpu::EltwiseOperation::Add, .broadcast = static_cast<fpu::SrcBBroadcast>(4), .release = fpu::SourceRelease::None}>();
}

//--- eltwise-address-modifier.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .address_modifier = 8, .release = fpu::SourceRelease::None}>();
}

//--- eltwise-dest-row-offset.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .dest_row_offset = 1024, .release = fpu::SourceRelease::None}>();
}

//--- matmul-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.matmul<fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::None}>();
}

//--- matmul-broadcast.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.matmul<fpu::Matmul {.broadcast = static_cast<fpu::SrcBRowBroadcast>(2), .release = fpu::SourceRelease::None}>();
}

//--- matmul-address-modifier.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.matmul<fpu::Matmul {.address_modifier = 8, .release = fpu::SourceRelease::None}>();
}

//--- matmul-row-broadcast-offset.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void issue()
{
    fpu::MatrixUnit.matmul<fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .dest_row_offset = 4, .release = fpu::SourceRelease::None}>();
}

//--- runtime-row-broadcast-offset.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

// Constant evaluation of the runtime encoder rejects the misaligned row broadcast without a static_assert.
constexpr auto operation =
    fpu::MatrixUnit.operation().matmul({.broadcast = fpu::SrcBRowBroadcast::Row, .dest_row_offset = 2, .release = fpu::SourceRelease::None});
// RUNTIME_ROW_BROADCAST: in 'constexpr' expansion of 'hal::detail::require({{.*}}"FPU matmul row broadcast needs a Dest row offset with bits 1 and 2 clear"))'
// RUNTIME_ROW_BROADCAST: error: '__builtin_trap()' is not a constant expression

//--- pool-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.pool<fpu::Pool {.function = fpu::PoolFunction::Maximum}>();
}

//--- pool-sum-indices.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.pool<fpu::Pool {.function = fpu::PoolFunction::Sum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None}>();
}

//--- pool-function.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::MatrixUnit.pool<fpu::Pool {.function = static_cast<fpu::PoolFunction>(2), .release = fpu::SourceRelease::None}>();
}
