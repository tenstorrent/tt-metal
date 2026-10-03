// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-cross-end.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_END
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-outside-start.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_START
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-source-size.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_SOURCE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-end.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-wrap.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-outside.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-read-end.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-read-wrap.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-read-outside.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-read-outside.cpp 2>&1 | FileCheck %s --check-prefix=GPR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-direct-state-outside-Bits32.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-direct-state-outside-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-direct-state-crossing-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-grouped-state-outside-Bits32.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-grouped-state-outside-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-grouped-state-crossing-Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-operand-outside.cpp 2>&1 | FileCheck %s --check-prefix=GPR_ADDR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-operand-span.cpp 2>&1 | FileCheck %s --check-prefix=GPR_SPAN
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-array-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_ARRAY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-read-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-gpr-direct-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GPR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-gpr-grouped-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GPR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-group-without-raw.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GROUP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-shifted-array-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_ARRAY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-shifted-read-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-section-array-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_ARRAY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-section-read-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_READ
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// ARRAY_END: error: static assertion failed: CFG array write crosses the end of the state bank
// ARRAY_START: error: static assertion failed: CFG array write starts outside the state bank
// ARRAY_SOURCE: error: static assertion failed: CFG word count exceeds source array
// READ_OFFSET: error: static assertion failed: CFG word offset crosses the selected bank
// GPR_READ: error: static assertion failed: CFG read source lies outside the state bank
// GPR_ADDR: error: static assertion failed: CFG write destination lies outside its register scope
// GPR_SPAN: error: static assertion failed: GPR write crosses the end of its CFG bank
// ANCHOR_ARRAY: error: static assertion failed: CFG array write extends past its anchor field
// ANCHOR_READ: error: static assertion failed: CFG word offset extends past its anchor field
// ANCHOR_GPR: error: static assertion failed: GPR write extends past its anchor field
// ANCHOR_GROUP: error: static assertion failed: whole-word CFG access requires a Field or a field group with a Raw anchor

// State: 40 + 184 reaches word 224; 40 + 0xffffffd8 would wrap to zero.
// Thread: 5 + 63 reaches word 68; 5 + 0xfffffffb would wrap to zero.

//--- fields.h
#pragma once

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Synthetic descriptors reach bank boundaries that no named field occupies.
inline constexpr cfg::Field state_first {cfg::RegisterScope::State, 32, 0, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_crossing {cfg::RegisterScope::State, 32, 222, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_outside {cfg::RegisterScope::State, 32, 224, 0, 0, 32, 1, 0};
inline constexpr cfg::Field thread_outside {cfg::RegisterScope::Thread, 16, 68, 0, 0, 16, 1, 0};
inline constexpr cfg::Field state_two_words {cfg::RegisterScope::State, 32, 64, 0, 0, 64, 1, 0};

// The same 64-bit field spans words 64-66 in S0 (bit 16) and 67-68 in S1 (bit 0).
inline constexpr cfg::Field state_sectioned_wide {cfg::RegisterScope::State, 32, 64, 0, 16, 64, 2, 80};

// A field group without a Raw anchor cannot stand in for a Field.
class GroupWithoutRaw
{
public:
    static constexpr cfg::Field Value {cfg::RegisterScope::State, 32, 64, 0, 0, 32, 1, 0};
};

inline constexpr GroupWithoutRaw group_without_raw {};

//--- array-cross-end.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 3>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::ChickenBits::sfpu_scbd_disable, cfg::Sec::S0, 3>(values);
}

//--- array-outside-start.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::MMIO, state_outside, cfg::Sec::S0, 1>(values);
}

//--- array-source-size.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::MMIO, state_first, cfg::Sec::S0, 2>(values);
}

//--- state-read-end.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, 184>();
}

//--- state-read-wrap.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, 0xffffffd8u>();
}

//--- state-read-outside.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, state_outside, cfg::Sec::S0>();
}

//--- thread-read-end.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, 63>();
}

//--- thread-read-wrap.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, 0xfffffffbu>();
}

//--- thread-read-outside.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, thread_outside, cfg::Sec::S0>();
}

//--- gpr-read-outside.cpp
#include "fields.h"

void probe()
{
    cfg::read<cfg::Access::TensixCfgUnit, state_outside, cfg::Sec::S0>(hal::gpr<4>());
}

//--- gpr-direct-state-outside-Bits32.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits32>(hal::gpr<4>());
}

//--- gpr-direct-state-outside-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- gpr-direct-state-crossing-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- gpr-grouped-state-outside-Bits32.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits32>(hal::gpr<4>()));
}

//--- gpr-grouped-state-outside-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_outside, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- gpr-grouped-state-crossing-Bits128.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- gpr-operand-outside.cpp
#include "fields.h"

void probe()
{
    auto operand = cfg::from_gpr<state_outside, cfg::Sec::S0>(hal::gpr<4>());
}

//--- gpr-operand-span.cpp
#include "fields.h"

void probe()
{
    auto operand = cfg::from_gpr<state_crossing, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- anchor-array-span.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 5>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 5>(values);
}

//--- anchor-read-span.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 4>();
}

//--- anchor-gpr-direct-span.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit, state_two_words, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- anchor-gpr-grouped-span.cpp
#include "fields.h"

void probe()
{
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::from_gpr<state_two_words, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- anchor-group-without-raw.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::MMIO, group_without_raw, cfg::Sec::S0, 1>(values);
}

//--- anchor-shifted-array-span.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 4>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 4>(values);
}

//--- anchor-shifted-read-span.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 3>();
}

//--- anchor-section-array-span.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 3>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 3>(values);
}

//--- anchor-section-read-span.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 2>();
}
