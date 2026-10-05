// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/constant-mmio.cpp 2>&1 | FileCheck %s --check-prefix=ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/constant-scalar.cpp 2>&1 | FileCheck %s --check-prefix=ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/runtime-scalar.cpp 2>&1 | FileCheck %s --check-prefix=VALUE_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/constant-wide.cpp 2>&1 | FileCheck %s --check-prefix=WIDE_WRITE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/runtime-wide.cpp 2>&1 | FileCheck %s --check-prefix=WIDE_WRITE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/constant-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/runtime-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/assignment-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/constant-value.cpp 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rmwcib-constant.cpp 2>&1 | FileCheck %s --check-prefix=RMWCIB_IGNORED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rmwcib-runtime.cpp 2>&1 | FileCheck %s --check-prefix=RMWCIB_IGNORED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rmwcib-constant-group.cpp 2>&1 | FileCheck %s --check-prefix=RMWCIB_IGNORED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rmwcib-runtime-group.cpp 2>&1 | FileCheck %s --check-prefix=RMWCIB_IGNORED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-scalar-constant.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-scalar-runtime.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-state-overlap.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-thread-overlap.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/assignment-wide-runtime.cpp 2>&1 | FileCheck %s --check-prefix=WIDE_ASSIGN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/assignment-wide-constant.cpp 2>&1 | FileCheck %s --check-prefix=WIDE_ASSIGN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-state-outside.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_OUTSIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-thread-outside.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_OUTSIDE

// ACCESS: error: static assertion failed: compile-time instruction emission requires Access::TensixCfgUnit
// VALUE_ACCESS: error: static assertion failed: value-backed cfg::write requires Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand
// WIDE_WRITE: error: static assertion failed: field wider than 32b cannot be written through a single value
// SECTION: error: static assertion failed: section index out of range for this register
// VALUE: error: static assertion failed: value exceeds field width
// RMWCIB_IGNORED: error: static assertion failed: RMWCIB writes to the state-reset register are ignored by hardware; use Access::MMIO or from_gpr
// GROUP_ACCESS: error: static assertion failed: field-assignment CFG writes require Access::MMIO or Access::TensixCfgUnit
// GROUP_OVERLAP: error: static assertion failed: overlapping CFG field assignments in one physical word
// WIDE_ASSIGN: error: static assertion failed: field wider than 32b cannot be assigned through a single value
// GROUP_OUTSIDE: error: static assertion failed: CFG write destination lies outside its register scope
// clang-format on

//--- fields.h
#pragma once

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Synthetic descriptors reach widths and addresses that no named field has.
inline constexpr cfg::Field state_wide {cfg::RegisterScope::State, 32, 64, 0, 0, 64, 1, 0};
inline constexpr cfg::Field state_outside {cfg::RegisterScope::State, 32, 224, 0, 0, 8, 1, 0};
inline constexpr cfg::Field thread_outside {cfg::RegisterScope::Thread, 16, 68, 0, 0, 16, 1, 0};

//--- constant-mmio.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::MMIO, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>();
}

//--- constant-scalar.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, 1>();
}

//--- runtime-scalar.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(value);
}

//--- constant-wide.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_wide, cfg::Sec::S0, 1>();
}

//--- runtime-wide.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit, state_wide, cfg::Sec::S0>(value);
}

//--- constant-section.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S1, 1>();
}

//--- runtime-section.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S1>(value);
}

//--- assignment-section.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S1>(value));
}

//--- constant-value.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 2>();
}

//--- rmwcib-constant.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::StateReset::EN, cfg::Sec::S0, 1>();
}

//--- rmwcib-runtime.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::StateReset::EN, cfg::Sec::S0>(value);
}

//--- rmwcib-constant-group.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::StateReset::EN, cfg::Sec::S0, 1>());
}

//--- rmwcib-runtime-group.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::StateReset::EN, cfg::Sec::S0>(value));
}

//--- group-scalar-constant.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit>(cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, 1>());
}

//--- group-scalar-runtime.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixScalarUnit>(cfg::set<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(value));
}

//--- group-state-overlap.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(), cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 0>());
}

//--- group-thread-overlap.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 1>(), cfg::set<cfg::SrcASet::Base, cfg::Sec::S0>(value));
}

//--- assignment-wide-runtime.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<state_wide, cfg::Sec::S0>(value));
}

//--- assignment-wide-constant.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<state_wide, cfg::Sec::S0, 1>());
}

//--- group-state-outside.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<state_outside, cfg::Sec::S0, 1>());
}

//--- group-thread-outside.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<thread_outside, cfg::Sec::S0>(value));
}
