// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template0-missing-start.cpp 2>&1 | FileCheck %s --check-prefix=MISSING_START
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template0-missing-start-shadow.cpp 2>&1 | FileCheck %s --check-prefix=MISSING_START_SHADOW
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template0-partial-mid-ops.cpp 2>&1 | FileCheck %s --check-prefix=PARTIAL_MID_OPS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template0-end-without-shadow.cpp 2>&1 | FileCheck %s --check-prefix=END_PAIR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template0-shadow-without-end.cpp 2>&1 | FileCheck %s --check-prefix=END_PAIR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template1-count-anomaly.cpp 2>&1 | FileCheck %s --check-prefix=COUNT_ANOMALY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template1-count-anomaly-aliased.cpp 2>&1 | FileCheck %s --check-prefix=COUNT_ANOMALY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-field-not-count.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_FIELD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-field-direct-not-count.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_PROGRAM
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-field-duplicate.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_UNIQUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-count-zero.cpp 2>&1 | FileCheck %s --check-prefix=RUNNER_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-count-too-large.cpp 2>&1 | FileCheck %s --check-prefix=RUNNER_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-runtime-mask-count-too-large.cpp 2>&1 | FileCheck %s --check-prefix=RUNNER_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-overrides-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=OVERRIDES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-overrides-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=OVERRIDES
// clang-format on

// MISSING_START: error: static assertion failed: Template 0 MOP requires start_op
// MISSING_START_SHADOW: error: static assertion failed: Template 0 MOP requires start_op_shadow
// PARTIAL_MID_OPS: error: static assertion failed: Template 0 MOP requires either all three mid ops or none
// END_PAIR: error: static assertion failed: Template 0 MOP requires end_op and end_op_shadow to be set together
// COUNT_ANOMALY: error: static assertion failed: Template 1 MOP configuration triggers the no-start/zero-inner/active-end count anomaly
// RUNTIME_FIELD: error: static assertion failed: Only Template 1 loop counts may be supplied with runtime_field
// RUNTIME_PROGRAM: error: static assertion failed: Only Template 1 loop counts may be supplied at runtime
// RUNTIME_UNIQUE: error: static assertion failed: Template 1 runtime fields must be unique
// RUNNER_COUNT: error: static assertion failed: MOP encoded count must fit in 7 bits
// OVERRIDES: error: static assertion failed: Template 1 MOP count overrides must fit in 10 bits

//--- configs.h
#pragma once

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

using Template0Config = mop::MopConfig<mop::MopTemplate::Template0>;
using Template1Config = mop::MopConfig<mop::MopTemplate::Template1>;

inline constexpr std::uint32_t OP = 0x10001000;

inline constexpr Template1Config with_start {
    .outer_loop = {.count = 1, .start_op = OP},
    .inner_loop = {.count = 1, .body_op = OP},
};

//--- template0-missing-start.cpp
#include "configs.h"

void probe()
{
    mop::program<Template0Config {.start_op_shadow = OP}>();
}

//--- template0-missing-start-shadow.cpp
#include "configs.h"

void probe()
{
    mop::program<Template0Config {.start_op = OP}>();
}

//--- template0-partial-mid-ops.cpp
#include "configs.h"

void probe()
{
    mop::program<Template0Config {.start_op = OP, .mid_ops = {.op_a = OP, .op_b = OP}, .start_op_shadow = OP}>();
}

//--- template0-end-without-shadow.cpp
#include "configs.h"

void probe()
{
    mop::program<Template0Config {.start_op = OP, .end_op = OP, .start_op_shadow = OP}>();
}

//--- template0-shadow-without-end.cpp
#include "configs.h"

void probe()
{
    mop::program<Template0Config {.start_op = OP, .start_op_shadow = OP, .end_op_shadow = OP}>();
}

//--- template1-count-anomaly.cpp
#include "configs.h"

void probe()
{
    mop::program<Template1Config {.outer_loop = {.count = 1, .end_op = OP}, .inner_loop = {.count = 0, .body_op = OP}}>();
}

//--- template1-count-anomaly-aliased.cpp
#include "configs.h"

// Only the low ten count bits are consumed, so 1024 is an effective zero.
void probe()
{
    mop::program<Template1Config {.outer_loop = {.count = 1, .end_op = OP}, .inner_loop = {.count = 1024, .body_op = OP}}>();
}

//--- runtime-field-not-count.cpp
#include "configs.h"

void probe(const std::uint32_t operation)
{
    mop::program<with_start>(mop::runtime_field<mop::Template1Field::InnerLoopBodyOp>(operation));
}

//--- runtime-field-direct-not-count.cpp
#include "configs.h"

void probe(const std::uint32_t operation)
{
    mop::program<with_start>(mop::RuntimeField<mop::Template1Field::InnerLoopBodyOp> {operation});
}

//--- runtime-field-duplicate.cpp
#include "configs.h"

void probe(const std::uint32_t a, const std::uint32_t b)
{
    mop::program<with_start>(mop::runtime_field<mop::Template1Field::InnerLoopCount>(a), mop::runtime_field<mop::Template1Field::InnerLoopCount>(b));
}

//--- runner-count-zero.cpp
#include "configs.h"

void probe()
{
    mop::Runner<mop::MopTemplate::Template0>::run<0, 1>();
}

//--- runner-count-too-large.cpp
#include "configs.h"

void probe()
{
    mop::Runner<mop::MopTemplate::Template0>::run<129, 1>();
}

//--- runner-runtime-mask-count-too-large.cpp
#include "configs.h"

void probe(const std::uint32_t mask)
{
    mop::Runner<mop::MopTemplate::Template0>::run_with_runtime_mask<129>(mask);
}

//--- runner-overrides-too-wide.cpp
#include "configs.h"

void probe()
{
    mop::Runner<mop::MopTemplate::Template1>::run<mop::Template1CountOverrides {.outer_loop_count = 1024}>();
}

//--- operation-overrides-too-wide.cpp
#include "configs.h"

std::uint32_t probe()
{
    return mop::get_operation<mop::Template1CountOverrides {.inner_loop_count = 1024}>();
}
