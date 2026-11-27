// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_unpack_thread}
// RUN: %split-file %s %t
// RUN: not %{diagnose} %t/thread-scope.cpp 2>&1 | FileCheck %s --check-prefix=STATE
// RUN: not %{diagnose} %t/partial-word.cpp 2>&1 | FileCheck %s --check-prefix=WHOLE
// RUN: not %{diagnose} %t/section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{diagnose} %t/outside.cpp 2>&1 | FileCheck %s --check-prefix=OUTSIDE

// STATE: error: static assertion failed: add_scratch requires a state-CFG destination
// WHOLE: error: static assertion failed: add_scratch requires a field covering a whole 32-bit CFG word
// SECTION: error: static assertion failed: section index out of range for this register
// OUTSIDE: error: static assertion failed: CFG write destination lies outside its register scope
// clang-format on

//--- fields.h
#pragma once

#include "hal/cfg.h"

namespace cfg = hal::cfg;

inline constexpr cfg::Field thread_word {cfg::RegisterScope::Thread, 16, 0, 0, 0, 16, 1, 0};
inline constexpr cfg::Field state_outside {cfg::RegisterScope::State, 32, 224, 0, 0, 32, 1, 0};

//--- thread-scope.cpp
#include "fields.h"

void test()
{
    cfg::add_scratch<thread_word, cfg::Sec::S0, cfg::ScratchSec::Thread>();
}

//--- partial-word.cpp
#include "fields.h"

void test()
{
    cfg::add_scratch<cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0, cfg::ScratchSec::Thread>();
}

//--- section.cpp
#include "fields.h"

void test()
{
    cfg::add_scratch<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S2, cfg::ScratchSec::Thread>();
}

//--- outside.cpp
#include "fields.h"

void test()
{
    cfg::add_scratch<state_outside, cfg::Sec::S0, cfg::ScratchSec::Thread>();
}
