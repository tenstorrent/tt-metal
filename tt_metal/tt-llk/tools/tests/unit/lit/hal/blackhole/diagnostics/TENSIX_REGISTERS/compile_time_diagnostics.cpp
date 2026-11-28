// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/valid-options.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/transpose-srca.cpp 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/transpose-both.cpp 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/gate-control-srcb.cpp 2>&1 | FileCheck %s --check-prefix=GATE_CONTROL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mark-invalid-index.cpp 2>&1 | FileCheck %s --check-prefix=MARK_INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mark-valid-face.cpp 2>&1 | FileCheck %s --check-prefix=MARK_VALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-index.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-face.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_FACE
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// TRANSPOSE: error: static assertion failed: transpose targets SrcB
// GATE_CONTROL: error: static assertion failed: GATESRCRST resets both gate controls: issue it on SrcA | SrcB
// MARK_INVALID: error: static assertion failed: invalid ZEROACC mark-invalid options
// MARK_VALID: error: static assertion failed: invalid ZEROACC mark-valid options

//--- valid-options.cpp
#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

// The row, face or half selector is the 10-bit Imm10.
static_assert(tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::SingleRow, .index = 1023, .address_mode = 7}));
static_assert(!tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::SingleRow, .index = 1024}));
static_assert(tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::Face, .index = 1023}));
static_assert(!tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::Face, .index = 1024}));
static_assert(tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::Half, .index = 1}));
static_assert(!tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::Half, .index = 2}));
static_assert(!tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::All, .index = 1}));
static_assert(!tr::is_valid(tr::DstMarkInvalid {.scope = tr::ZeroScope::All, .address_mode = 8}));
static_assert(!tr::is_valid(tr::DstMarkInvalid {.scope = static_cast<tr::ZeroScope>(4)}));
static_assert(tr::is_valid(tr::DstMarkValid {.face = 1023, .width = tr::DestWidth::Bits16, .address_mode = 7}));
static_assert(!tr::is_valid(tr::DstMarkValid {.face = 1024}));
static_assert(!tr::is_valid(tr::DstMarkValid {.address_mode = 8}));

// The runtime encoders are also usable in constant expressions.
static_assert(
    tr::Dst.operation().mark_invalid({.scope = tr::ZeroScope::Half, .index = 1, .width = tr::DestWidth::Bits32}) ==
    tr::Dst.operation().mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::Half, .index = 1, .width = tr::DestWidth::Bits32}>());
static_assert(tr::Dst.operation().mark_valid({.face = 7}) == tr::Dst.operation().mark_valid<tr::DstMarkValid {.face = 7}>());

//--- transpose-srca.cpp
#include "hal/tensix_registers.h"

void f()
{
    hal::tensix_registers::SrcA.transpose();
}

//--- transpose-both.cpp
#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

void f()
{
    (tr::SrcA | tr::SrcB).transpose();
}

//--- gate-control-srcb.cpp
#include "hal/tensix_registers.h"

void f()
{
    hal::tensix_registers::SrcB.reset_gate_control();
}

//--- mark-invalid-index.cpp
#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

void f()
{
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::SingleRow, .index = 1024}>();
}

//--- mark-valid-face.cpp
#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

void f()
{
    tr::Dst.mark_valid<tr::DstMarkValid {.face = 1024}>();
}

//--- runtime-index.cpp
#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

// Constant evaluation of a runtime encoder rejects invalid options without a static_assert.
constexpr auto operation = tr::Dst.operation().mark_invalid({.scope = tr::ZeroScope::All, .index = 1});
// RUNTIME_INDEX: in 'constexpr' expansion of 'hal::detail::require({{.*}}"ZEROACC index must be in [0, 1023], 0 or 1 for Half, and 0 for All"))'
// RUNTIME_INDEX: error: '__builtin_trap()' is not a constant expression

//--- runtime-face.cpp
#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

constexpr auto operation = tr::Dst.operation().mark_valid({.face = 1024});
// RUNTIME_FACE: in 'constexpr' expansion of 'hal::detail::require({{.*}}"ZEROACC face must be in [0, 1023]"))'
// RUNTIME_FACE: error: '__builtin_trap()' is not a constant expression
