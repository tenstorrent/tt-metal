// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/elementwise-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=ELEMENTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/elementwise-operation.cpp 2>&1 | FileCheck %s --check-prefix=ELEMENTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/elementwise-broadcast.cpp 2>&1 | FileCheck %s --check-prefix=ELEMENTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/elementwise-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=ELEMENTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/elementwise-dest-row-offset.cpp 2>&1 | FileCheck %s --check-prefix=ELEMENTWISE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/dot-product-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=DOT_PRODUCT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/dot-product-dest-row-offset.cpp 2>&1 | FileCheck %s --check-prefix=DOT_PRODUCT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matrix-multiply-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=MATRIX_MULTIPLY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matrix-multiply-broadcast.cpp 2>&1 | FileCheck %s --check-prefix=MATRIX_MULTIPLY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/matrix-multiply-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=MATRIX_MULTIPLY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/pool-unset-release.cpp 2>&1 | FileCheck %s --check-prefix=POOL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/pool-sum-indices.cpp 2>&1 | FileCheck %s --check-prefix=POOL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/pool-function.cpp 2>&1 | FileCheck %s --check-prefix=POOL
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// ELEMENTWISE: error: static assertion failed: invalid elementwise descriptor
// DOT_PRODUCT: error: static assertion failed: invalid dot-product descriptor
// MATRIX_MULTIPLY: error: static assertion failed: invalid matrix-multiply descriptor
// POOL: error: static assertion failed: invalid pool descriptor

//--- elementwise-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add}>();
}

//--- elementwise-operation.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Elementwise {.operation = static_cast<fpu::ElementwiseOperation>(3), .release = fpu::SourceRelease::None}>();
}

//--- elementwise-broadcast.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Elementwise {
        .operation = fpu::ElementwiseOperation::Add, .broadcast = static_cast<fpu::SrcBBroadcast>(4), .release = fpu::SourceRelease::None}>();
}

//--- elementwise-address-modifier.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .address_modifier = 8, .release = fpu::SourceRelease::None}>();
}

//--- elementwise-dest-row-offset.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .dest_row_offset = 1024, .release = fpu::SourceRelease::None}>();
}

//--- dot-product-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::DotProduct {.address_modifier = 0}>();
}

//--- dot-product-dest-row-offset.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::DotProduct {.dest_row_offset = 1024, .release = fpu::SourceRelease::None}>();
}

//--- matrix-multiply-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::MatrixMultiply {.broadcast = fpu::SrcBRowBroadcast::None}>();
}

//--- matrix-multiply-broadcast.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::MatrixMultiply {.broadcast = static_cast<fpu::SrcBRowBroadcast>(2), .release = fpu::SourceRelease::None}>();
}

//--- matrix-multiply-address-modifier.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::MatrixMultiply {.address_modifier = 8, .release = fpu::SourceRelease::None}>();
}

//--- pool-unset-release.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum}>();
}

//--- pool-sum-indices.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None}>();
}

//--- pool-function.cpp
#include "hal/fpu.h"

namespace fpu = hal::fpu;

void f()
{
    fpu::run<fpu::Pool {.function = static_cast<fpu::PoolFunction>(2), .release = fpu::SourceRelease::None}>();
}
