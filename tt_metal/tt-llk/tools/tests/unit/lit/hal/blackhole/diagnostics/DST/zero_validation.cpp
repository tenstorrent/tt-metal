// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_diagnose} %{blackhole_math_thread} %s

#include "hal/dst.h"

namespace dst = hal::dst;

static_assert(dst::is_valid(dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = (1u << 14) - 1, .address_mode = 7}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 1u << 14}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::SingleRow, .address_mode = 8}));
static_assert(dst::is_valid(dst::Zero {.scope = dst::ZeroScope::Face, .index = 255}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::Face, .index = 256}));
static_assert(dst::is_valid(dst::Zero {.scope = dst::ZeroScope::Half, .index = 1}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::Half, .index = 2}));
static_assert(dst::is_valid(dst::Zero {.scope = dst::ZeroScope::All, .flags = dst::ZeroFlagAction::ClearFlags, .width = dst::DestWidth::Bits32}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::All, .index = 1}));
static_assert(!dst::is_valid(dst::Zero {.scope = static_cast<dst::ZeroScope>(4)}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::All, .flags = static_cast<dst::ZeroFlagAction>(2)}));
static_assert(!dst::is_valid(dst::Zero {.scope = dst::ZeroScope::All, .width = static_cast<dst::DestWidth>(2)}));
