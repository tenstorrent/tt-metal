// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-section.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-value.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_VALUE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-packer.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_PACKER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-unpacker.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_UNPACKER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-mapping.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_MAPPING
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-access.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-gpr-destination.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_GPR_DESTINATION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/misaligned-wrcfg.cpp 2>&1 | FileCheck %s --check-prefix=MISALIGNED_WRCFG
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/misaligned-reg2flop.cpp 2>&1 | FileCheck %s --check-prefix=MISALIGNED_REG2FLOP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-scalar-destination.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_SCALAR_DESTINATION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/overlapping-assignments.cpp 2>&1 | FileCheck %s --check-prefix=OVERLAPPING_ASSIGNMENTS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/overlapping-operations.cpp 2>&1 | FileCheck %s --check-prefix=OVERLAPPING_OPERATIONS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/rmwcib-ignored.cpp 2>&1 | FileCheck %s --check-prefix=RMWCIB_IGNORED
// clang-format on

//--- invalid-section.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_section = cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S1, 1>();
// INVALID_SECTION: error: static assertion failed: section index out of range for this register

//--- invalid-value.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_value = cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 2>();
// INVALID_VALUE: error: static assertion failed: value exceeds field width

//--- invalid-packer.cpp
#include <cstdint>
#include <type_traits>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_packer = cfg::Packer[std::integral_constant<std::uint32_t, 1> {}];
// INVALID_PACKER: error: static assertion failed: Blackhole exposes address configuration only for packer 0

//--- invalid-unpacker.cpp
#include <cstdint>
#include <type_traits>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_unpacker = cfg::Unpacker[std::integral_constant<std::uint32_t, 2> {}];
// INVALID_UNPACKER: error: static assertion failed: unpacker index out of range

//--- invalid-context.cpp
#include <cstdint>
#include <type_traits>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_context = cfg::Unpacker[std::integral_constant<std::uint32_t, 1> {}].Cntx[std::integral_constant<std::uint32_t, 2> {}];
// INVALID_CONTEXT: error: static assertion failed: unpacker context index out of range

//--- invalid-mapping.cpp
#include <cstdint>
#include <type_traits>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_mapping = cfg::TileRowSetMapping[std::integral_constant<std::uint32_t, 4> {}];
// INVALID_MAPPING: error: static assertion failed: tile row-set mapping index out of range

//--- invalid-access.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void invalid_access()
{
    cfg::write<cfg::Access::MMIO, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>();
    // INVALID_ACCESS: error: static assertion failed: compile-time instruction emission requires Access::TensixCfgUnit
}

//--- invalid-gpr-destination.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

constexpr auto invalid_gpr_destination = cfg::from_gpr<cfg::CfgStateId::StateID, cfg::Sec::S0>(cfg::gpr<4>());
// INVALID_GPR_DESTINATION: error: static assertion failed: GPR-backed CFG writes require a state-CFG destination

//--- misaligned-wrcfg.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void misaligned_wrcfg()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(cfg::gpr<8, cfg::GprTransferSize::Bits128>());
    // MISALIGNED_WRCFG: error: static assertion failed: 128-bit GPR cfg::write destination must be four-word aligned
}

//--- misaligned-reg2flop.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void misaligned_reg2flop()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S0>(cfg::gpr<10, cfg::GprTransferSize::Bits128>());
    // MISALIGNED_REG2FLOP: error: static assertion failed: 128-bit REG2FLOP source GPR must be four-word aligned
}

//--- invalid-scalar-destination.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void invalid_scalar_destination()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr<4>());
    // INVALID_SCALAR_DESTINATION: error: static assertion failed: Access::TensixScalarUnit supports THCON CFG destinations only
}

//--- overlapping-assignments.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void overlapping_assignments()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(), cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 0>());
    // OVERLAPPING_ASSIGNMENTS: error: static assertion failed: overlapping CFG field assignments in one physical word
}

//--- overlapping-operations.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void overlapping_operations()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 1>(), cfg::from_gpr<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0>(hal::gpr<4>()));
    // OVERLAPPING_OPERATIONS: error: static assertion failed: overlapping field assignments or GPR destination spans in cfg::write
}

//--- rmwcib-ignored.cpp
#include "hal/cfg.h"

namespace cfg = hal::cfg;

void rmwcib_ignored()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::StateReset::EN, cfg::Sec::S0, 1>();
    // RMWCIB_IGNORED: error: static assertion failed: RMWCIB writes to the state-reset register are ignored by hardware
}
