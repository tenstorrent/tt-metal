// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// Check all four occupied words, with the GPR transfer before and after the field.
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=0 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=1 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=2 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=3 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=0 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=1 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=2 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field.cpp -DCFG_TEST_WORD=3 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-gpr.cpp 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-gpr.cpp -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-fields.cpp 2>&1 | FileCheck %s --check-prefix=FIELDS
// clang-format on

// MIXED: error: static assertion failed: overlapping field assignments or GPR destination spans in cfg::write
// FIELDS: error: static assertion failed: overlapping CFG field assignments in one physical word

//--- gpr-field.cpp
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

inline constexpr cfg::Field overlapping {cfg::RegisterScope::State, 32, 76 + CFG_TEST_WORD, 0, 0, 1, 1, 0};

void probe(std::uint32_t value)
{
    const auto transfer = cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
    const auto field    = cfg::set<overlapping, cfg::Sec::S0>(value);
#ifdef CFG_TEST_REVERSE
    cfg::write<cfg::Access::TensixCfgUnit>(field, transfer);
#else
    cfg::write<cfg::Access::TensixCfgUnit>(transfer, field);
#endif
}

//--- gpr-gpr.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void probe()
{
    const auto wide   = cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
    const auto narrow = cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_cntx3_address, cfg::Sec::S0>(hal::gpr<8>());
#ifdef CFG_TEST_REVERSE
    cfg::write<cfg::Access::TensixCfgUnit>(narrow, wide);
#else
    cfg::write<cfg::Access::TensixCfgUnit>(wide, narrow);
#endif
}

//--- thread-fields.cpp
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 1>(), cfg::set<cfg::SrcASet::Base, cfg::Sec::S0>(value));
}
