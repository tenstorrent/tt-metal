// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-reserved.cpp 2>&1 | FileCheck %s --check-prefix=GPR_RESERVED
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-mmio.cpp 2>&1 | FileCheck %s --check-prefix=RDCFG_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-scalar.cpp 2>&1 | FileCheck %s --check-prefix=RDCFG_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-runtime-index.cpp 2>&1 | FileCheck %s --check-prefix=RDCFG_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-thread.cpp 2>&1 | FileCheck %s --check-prefix=RDCFG_THREAD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-wide.cpp 2>&1 | FileCheck %s --check-prefix=RDCFG_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-word-crossing.cpp 2>&1 | FileCheck %s --check-prefix=RDCFG_CROSS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/from-gpr-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/rdcfg-outside.cpp 2>&1 | FileCheck %s --check-prefix=GPR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/write-mmio.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-mmio.cpp 2>&1 | FileCheck %s --check-prefix=HETEROGENEOUS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-mmio-mixed.cpp 2>&1 | FileCheck %s --check-prefix=HETEROGENEOUS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-scalar.cpp 2>&1 | FileCheck %s --check-prefix=HETEROGENEOUS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/from-gpr-thread.cpp 2>&1 | FileCheck %s --check-prefix=GPR_STATE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/write-thread.cpp 2>&1 | FileCheck %s --check-prefix=GPR_STATE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/from-gpr-unaligned-field.cpp 2>&1 | FileCheck %s --check-prefix=GPR_START
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/write-unaligned-field.cpp 2>&1 | FileCheck %s --check-prefix=GPR_START
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/direct-outside-Bits32.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/direct-outside-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/grouped-outside-Bits32.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/grouped-outside-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/operand-outside.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/direct-crossing-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/grouped-crossing-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/operand-crossing-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-direct-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GPR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-grouped-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GPR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-group-without-raw.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GROUP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/wrcfg-misaligned.cpp 2>&1 | FileCheck %s --check-prefix=WRCFG_ALIGN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/wrcfg-grouped-misaligned.cpp 2>&1 | FileCheck %s --check-prefix=WRCFG_ALIGN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/scalar-destination.cpp 2>&1 | FileCheck %s --check-prefix=SCALAR_DEST
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/scalar-index.cpp 2>&1 | FileCheck %s --check-prefix=SCALAR_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/scalar-source-misaligned.cpp 2>&1 | FileCheck %s --check-prefix=SCALAR_SOURCE_ALIGN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/scalar-span.cpp 2>&1 | FileCheck %s --check-prefix=WRCFG_ALIGN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/scalar-span.cpp 2>&1 | FileCheck %s --check-prefix=SCALAR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/field-overlap.cpp 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=0 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=1 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=2 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=3 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=0 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=1 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=2 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-field-overlap.cpp -DCFG_TEST_WORD=3 -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-gpr-overlap.cpp 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-gpr-overlap.cpp -DCFG_TEST_REVERSE 2>&1 | FileCheck %s --check-prefix=OVERLAP

// Every four-word aligned THCON start fits (176 + 3 < 180), so a 128-bit
// REG2FLOP that crosses the THCON range end is also misaligned.

// GPR_RESERVED: error: static assertion failed: GPR index is reserved by hal::gpr()
// RDCFG_ACCESS: error: static assertion failed: RDCFG requires Access::TensixCfgUnit
// RDCFG_INDEX: error: static assertion failed: RDCFG requires a compile-time GPR index: use hal::gpr<Index>()
// RDCFG_THREAD: error: static assertion failed: RDCFG cannot read thread CFG (SETC16) fields
// RDCFG_WIDE: error: static assertion failed: field wider than 32b cannot be selected through a single CFG word
// RDCFG_CROSS: error: static assertion failed: field crosses a CFG word boundary
// SECTION: error: static assertion failed: section index out of range for this register
// GPR_READ: error: static assertion failed: CFG read source lies outside the state bank
// GPR_ACCESS: error: static assertion failed: GPR-backed cfg::write requires Access::TensixCfgUnit or Access::TensixScalarUnit
// HETEROGENEOUS: error: static assertion failed: heterogeneous cfg::write supports Access::TensixCfgUnit only
// GPR_STATE: error: static assertion failed: GPR-backed CFG writes require a state-CFG destination
// GPR_START: error: static assertion failed: GPR-backed CFG writes must start at the beginning of a CFG word
// GPR_ADDR: error: static assertion failed: CFG write destination lies outside its register scope
// GPR_SPAN: error: static assertion failed: GPR write crosses the end of its CFG bank
// ANCHOR_GPR: error: static assertion failed: GPR write extends past its anchor field
// ANCHOR_GROUP: error: static assertion failed: whole-word CFG access requires a Field or a field group with a Raw anchor
// WRCFG_ALIGN: error: static assertion failed: 128-bit GPR cfg::write destination must be four-word aligned
// SCALAR_DEST: error: static assertion failed: Access::TensixScalarUnit supports THCON CFG destinations only
// SCALAR_INDEX: error: static assertion failed: REG2FLOP GPR index must be in [0, 63]
// SCALAR_SOURCE_ALIGN: error: static assertion failed: 128-bit REG2FLOP source GPR must be four-word aligned
// SCALAR_SPAN: error: static assertion failed: 128-bit REG2FLOP transfer crosses the THCON CFG range
// OVERLAP: error: static assertion failed: overlapping field assignments or GPR destination spans in cfg::write
// clang-format on

//--- fields.h
#pragma once

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Synthetic descriptors reach widths, alignments, and bank boundaries that no named field has.
inline constexpr cfg::Field state_crossing {cfg::RegisterScope::State, 32, 222, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_outside {cfg::RegisterScope::State, 32, 224, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_two_words {cfg::RegisterScope::State, 32, 64, 0, 0, 64, 1, 0};
inline constexpr cfg::Field state_word_crossing {cfg::RegisterScope::State, 32, 64, 0, 28, 8, 1, 0};
inline constexpr cfg::Field thcon_crossing {cfg::RegisterScope::State, 32, 177, 0, 0, 32, 1, 0};

// A field group without a Raw anchor cannot stand in for a Field.
class GroupWithoutRaw
{
public:
    static constexpr cfg::Field Value {cfg::RegisterScope::State, 32, 64, 0, 0, 32, 1, 0};
};

inline constexpr GroupWithoutRaw group_without_raw {};

//--- gpr-reserved.cpp
#include "fields.h"

constexpr auto reserved = hal::gpr<0xffffffffu>();

//--- rdcfg-mmio.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::MMIO, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr<4>());
}

//--- rdcfg-scalar.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixScalarUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr<4>());
}

//--- rdcfg-runtime-index.cpp
#include "fields.h"

void probe(std::uint32_t index)
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr(index));
}

//--- rdcfg-thread.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::SrcASet::Base, cfg::Sec::S0>(hal::gpr<4>());
}

//--- rdcfg-wide.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixCfgUnit, state_two_words, cfg::Sec::S0>(hal::gpr<4>());
}

//--- rdcfg-word-crossing.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixCfgUnit, state_word_crossing, cfg::Sec::S0>(hal::gpr<4>());
}

//--- rdcfg-section.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S1>(hal::gpr<4>());
}

//--- from-gpr-section.cpp
#include "fields.h"

constexpr auto operand = cfg::from_gpr<cfg::PrngSeed::Seed_Val, cfg::Sec::S1>(hal::gpr<4>());

//--- rdcfg-outside.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixCfgUnit, state_outside, cfg::Sec::S0>(hal::gpr<4>());
}

//--- write-mmio.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>());
}

//--- group-mmio.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::MMIO>(cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()));
}

//--- group-mmio-mixed.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(), cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()));
}

//--- group-scalar.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit>(cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()));
}

//--- from-gpr-thread.cpp
#include "fields.h"

constexpr auto operand = cfg::from_gpr<cfg::CfgStateId::StateID, cfg::Sec::S0>(hal::gpr<4>());

//--- write-thread.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::SrcASet::Base, cfg::Sec::S0>(hal::gpr<4>());
}

//--- from-gpr-unaligned-field.cpp
#include "fields.h"

constexpr auto operand = cfg::from_gpr<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0>(hal::gpr<4>());

//--- write-unaligned-field.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0>(hal::gpr<4>());
}

//--- direct-outside-Bits32.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits32>(hal::gpr<4>());
}

//--- direct-outside-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- grouped-outside-Bits32.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits32>(hal::gpr<4>()));
}

//--- grouped-outside-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- operand-outside.cpp
#include "fields.h"

constexpr auto operand = cfg::from_gpr<state_outside, cfg::Sec::S0>(hal::gpr<4>());

//--- direct-crossing-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- grouped-crossing-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- operand-crossing-Bits128.cpp
#include "fields.h"

constexpr auto operand = cfg::from_gpr<state_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());

//--- anchor-direct-span.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_two_words, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- anchor-grouped-span.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_two_words, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- anchor-group-without-raw.cpp
#include "fields.h"

constexpr auto operand = cfg::from_gpr<group_without_raw, cfg::Sec::S0>(hal::gpr<4>());

//--- wrcfg-misaligned.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<8>());
}

//--- wrcfg-grouped-misaligned.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<8>()));
}

//--- scalar-destination.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>(hal::gpr<4>());
}

//--- scalar-index.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<64>());
}

//--- scalar-source-misaligned.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit, cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<10>());
}

//--- scalar-span.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixScalarUnit, thcon_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- field-overlap.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 1>(), cfg::from_gpr<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0>(hal::gpr<4>()));
}

//--- gpr-field-overlap.cpp
#include "fields.h"

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

//--- gpr-gpr-overlap.cpp
#include "fields.h"

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
