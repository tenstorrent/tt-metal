// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_diagnose} %{blackhole_math_thread} %s

#include "hal/fpu.h"

namespace fpu = hal::fpu;

static_assert(fpu::is_valid(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add}));
static_assert(fpu::is_valid(fpu::Elementwise {
    .operation        = fpu::ElementwiseOperation::Multiply,
    .broadcast        = fpu::SrcBBroadcast::Scalar,
    .dest_write       = fpu::DestWriteMode::Accumulate,
    .address_modifier = 7,
    .dest_row_offset  = 1023,
    .release          = fpu::SourceRelease::Both}));
static_assert(!fpu::is_valid(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .address_modifier = 8, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .dest_row_offset = 1024, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Elementwise {
    .operation = fpu::ElementwiseOperation::Add, .dest_write = static_cast<fpu::DestWriteMode>(2), .release = fpu::SourceRelease::None}));

static_assert(fpu::is_valid(fpu::DotProduct {.release = fpu::SourceRelease::SrcA}));
static_assert(!fpu::is_valid(fpu::DotProduct {}));
static_assert(!fpu::is_valid(fpu::DotProduct {.release = static_cast<fpu::SourceRelease>(4)}));

static_assert(fpu::is_valid(fpu::MatrixMultiply {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::SrcB}));
static_assert(!fpu::is_valid(fpu::MatrixMultiply {}));
static_assert(!fpu::is_valid(fpu::MatrixMultiply {.dest_row_offset = 1024, .release = fpu::SourceRelease::None}));

static_assert(fpu::is_valid(fpu::Pool {.function = fpu::PoolFunction::Maximum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Pool {.function = fpu::PoolFunction::Sum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Pool {.function = fpu::PoolFunction::Sum}));
static_assert(!fpu::is_valid(fpu::Pool {
    .function = fpu::PoolFunction::Sum, .indices = static_cast<fpu::IndexTracking>(2), .release = fpu::SourceRelease::None}));
