// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/if-clear-three-ops.cpp 2>&1 | FileCheck %s --check-prefix=IF_CLEAR_SHAPE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/if-clear-six-ops.cpp 2>&1 | FileCheck %s --check-prefix=IF_CLEAR_SHAPE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/b-without-skip-b.cpp 2>&1 | FileCheck %s --check-prefix=B_PAIR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/skip-b-without-b.cpp 2>&1 | FileCheck %s --check-prefix=B_PAIR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template1-count-anomaly.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template1-count-anomaly-second-end.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template1-outer-count-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/template1-inner-count-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/patch-empty-inner.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/patch-not-last-op.cpp 2>&1 | FileCheck %s --check-prefix=PATCH_FIELD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/update-empty.cpp 2>&1 | FileCheck %s --check-prefix=UPDATE_EMPTY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/update-duplicate.cpp 2>&1 | FileCheck %s --check-prefix=UPDATE_UNIQUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-count-zero.cpp 2>&1 | FileCheck %s --check-prefix=RUNNER_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-count-too-large.cpp 2>&1 | FileCheck %s --check-prefix=RUNNER_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runner-runtime-mask-count-too-large.cpp 2>&1 | FileCheck %s --check-prefix=RUNNER_COUNT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-count-zero.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-long-count.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/operation-wide-mask.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT
// clang-format on

// IF_CLEAR_SHAPE: error: static assertion failed: Template 0 MOP if_clear takes {a0}, {a0, b}, {a0, a1, a2, a3} or {a0, a1, a2, a3, b}
// B_PAIR: error: static assertion failed: Template 0 MOP issues InsnB and SkipB under one flag
// PATCH_FIELD: error: static assertion failed: Template 1 MOP patch replaces last_outer or last_inner
// UPDATE_EMPTY: error: static assertion failed: Template 1 MOP update takes at least one word
// UPDATE_UNIQUE: error: static assertion failed: Template 1 MOP update takes each word at most once
// RUNNER_COUNT: error: static assertion failed: MOP encoded count must fit in 7 bits

// Checks evaluated in a constexpr configuration trap instead of reporting their message.
// CONSTANT: error: {{.*}}not a constant expression
// CONSTANT: __builtin_trap

//--- configs.h
#pragma once

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

inline constexpr std::uint32_t OP = 0x10001000;

//--- if-clear-three-ops.cpp
#include "configs.h"

constexpr auto bad = mop::if_clear(OP, OP, OP);

//--- if-clear-six-ops.cpp
#include "configs.h"

constexpr auto bad = mop::if_clear(OP, OP, OP, OP, OP, OP);

//--- b-without-skip-b.cpp
#include "configs.h"

constexpr auto bad = mop::foreach_mask_bit(mop::if_clear(OP, OP), mop::otherwise(OP));

//--- skip-b-without-b.cpp
#include "configs.h"

constexpr auto bad = mop::foreach_mask_bit(mop::if_clear(OP), mop::otherwise(OP, OP));

//--- template1-count-anomaly.cpp
#include "configs.h"

constexpr auto bad = mop::loop(4, mop::loop(0, OP), OP);

//--- template1-count-anomaly-second-end.cpp
#include "configs.h"

// With EndOp1 set the anomaly still triggers when the outer count is 1.
constexpr auto bad = mop::loop(1, mop::loop(0, OP), OP, OP);

//--- template1-outer-count-too-wide.cpp
#include "configs.h"

constexpr auto bad = mop::loop(1024, mop::loop(1, OP));

//--- template1-inner-count-too-wide.cpp
#include "configs.h"

constexpr auto bad = mop::loop(1, mop::loop(1024, OP));

//--- patch-empty-inner.cpp
#include "configs.h"

constexpr auto bad = mop::loop(1, OP, mop::loop(0, OP)).patch(mop::last_outer, OP);

//--- patch-not-last-op.cpp
#include "configs.h"

constexpr auto bad = mop::loop(1, mop::loop(1, OP)).patch(mop::Field<5> {}, OP);

//--- update-empty.cpp
#include "configs.h"

void probe()
{
    mop::update();
}

//--- update-duplicate.cpp
#include "configs.h"

void probe(const std::uint32_t a, const std::uint32_t b)
{
    mop::update(mop::inner_count(a), mop::inner_count(b));
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

//--- operation-count-zero.cpp
#include "configs.h"

constexpr std::uint32_t bad = mop::Runner<mop::MopTemplate::Template0> {.count = 0}.operation();

//--- operation-long-count.cpp
#include "configs.h"

// With MaskHigh::Auto the encoding cannot carry the MOP_CFG a run of 17 needs.
constexpr std::uint32_t bad = mop::Runner<mop::MopTemplate::Template0> {.count = 17}.operation();

//--- operation-wide-mask.cpp
#include "configs.h"

constexpr std::uint32_t bad = mop::Runner<mop::MopTemplate::Template0> {.count = 1, .mask = 0x10000}.operation();
