// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_diagnose} %{blackhole_math_thread} %s

#include "hal/fpu.h"

namespace fpu = hal::fpu;

static_assert(fpu::is_valid(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add}));
static_assert(fpu::is_valid(fpu::Eltwise {
    .operation        = fpu::EltwiseOperation::Multiply,
    .broadcast        = fpu::SrcBBroadcast::Scalar,
    .dest_write       = fpu::DestWriteMode::Accumulate,
    .address_modifier = 7,
    .dest_row_offset  = 1023,
    .release          = fpu::SourceRelease::Both}));
static_assert(!fpu::is_valid(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .address_modifier = 8, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .dest_row_offset = 1024, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Eltwise {
    .operation = fpu::EltwiseOperation::Add, .dest_write = static_cast<fpu::DestWriteMode>(2), .release = fpu::SourceRelease::None}));

static_assert(fpu::is_valid(fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::SrcB}));
static_assert(!fpu::is_valid(fpu::Matmul {}));
static_assert(!fpu::is_valid(fpu::Matmul {.dest_row_offset = 1024, .release = fpu::SourceRelease::None}));
// Row broadcast with Dst row bit 1 or 2 set is UnsupportedFunctionality.
static_assert(fpu::is_valid(fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .dest_row_offset = 8, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .dest_row_offset = 2, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .dest_row_offset = 4, .release = fpu::SourceRelease::None}));
static_assert(fpu::is_valid(fpu::Matmul {.dest_row_offset = 2, .release = fpu::SourceRelease::None}));

static_assert(fpu::is_valid(fpu::Pool {.function = fpu::PoolFunction::Maximum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Pool {.function = fpu::PoolFunction::Sum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None}));
static_assert(!fpu::is_valid(fpu::Pool {.function = fpu::PoolFunction::Sum}));
static_assert(!fpu::is_valid(fpu::Pool {
    .function = fpu::PoolFunction::Sum, .indices = static_cast<fpu::IndexTracking>(2), .release = fpu::SourceRelease::None}));
