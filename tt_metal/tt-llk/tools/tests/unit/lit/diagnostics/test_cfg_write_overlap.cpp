// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: blackhole-cfg, sfpi
// DEFINE: %{check} = %{sfpi_cxx} %{cfg_flags} -DCOMPILE_FOR_TRISC=0 -fsyntax-only -fmax-errors=3 -I %S/../Inputs
// RUN: split-file %s %t
//
// Check all four occupied words, with the GPR transfer before and after the field.
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=0 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=1 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=2 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=3 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=0 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=1 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=2 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_field.cpp -DCFG_TEST_WORD=3 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_gpr.cpp 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/gpr_gpr.cpp -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{check} %t/fields.cpp 2>&1 | FileCheck %s --check-prefix=FIELDS
//
// MIXED: error: static assertion failed: overlapping field assignments or GPR destination spans in cfg::write
// FIELDS: error: static assertion failed: overlapping CFG field assignments in one physical word

//--- gpr_field.cpp
#include <cstdint>

#include "cfg_test_fields.h"

inline constexpr Field overlapping {RegisterScope::State, 32, 76 + CFG_TEST_WORD, 0, 0, 1, 1, 0};

void probe(std::uint32_t value)
{
    const auto transfer = from_gpr<ThconReg3Fields::Base_address, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
    const auto field    = set<overlapping, Sec::S0>(value);
#ifdef CFG_TEST_REVERSE
    write<Access::TensixCfgUnit>(field, transfer);
#else
    write<Access::TensixCfgUnit>(transfer, field);
#endif
}

//--- gpr_gpr.cpp
#include "cfg_test_fields.h"

inline constexpr Field last_word {RegisterScope::State, 32, 79, 0, 0, 32, 1, 0};

void probe()
{
    const auto wide   = from_gpr<ThconReg3Fields::Base_address, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
    const auto narrow = from_gpr<last_word, Sec::S0>(hal::gpr<8>());
#ifdef CFG_TEST_REVERSE
    write<Access::TensixCfgUnit>(narrow, wide);
#else
    write<Access::TensixCfgUnit>(wide, narrow);
#endif
}

//--- fields.cpp
#include "cfg_test_fields.h"

void probe(std::uint32_t value)
{
    write<Access::TensixCfgUnit>(set<SrcASet::Base, Sec::S0, 1>(), set<SrcASet::Base, Sec::S0>(value));
}
