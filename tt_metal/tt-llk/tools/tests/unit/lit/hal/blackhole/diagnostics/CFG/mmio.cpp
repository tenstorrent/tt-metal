// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/read-word-tensix.cpp 2>&1 | FileCheck %s --check-prefix=READ_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/read-field-tensix.cpp 2>&1 | FileCheck %s --check-prefix=READ_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/read-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/write-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-end.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-wrap.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-outside.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-read-end.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-read-wrap.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/thread-read-outside.cpp 2>&1 | FileCheck %s --check-prefix=READ_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-read-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-shifted-read-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-section-read-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/state-read-target.cpp 2>&1 | FileCheck %s --check-prefix=TARGET_STATE
// RUN: not %{blackhole_tensix_diagnose} %t/thread-read-brisc.cpp 2>&1 | FileCheck %s --check-prefix=BRISC_TARGET
// RUN: not %{blackhole_tensix_diagnose} %t/thread-read-brisc.cpp -DCOMPILE_FOR_TRISC=3 2>&1 | FileCheck %s --check-prefix=TRISC_ID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/read-wide.cpp 2>&1 | FileCheck %s --check-prefix=EXTRACT_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/extract-wide.cpp 2>&1 | FileCheck %s --check-prefix=EXTRACT_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/read-word-crossing.cpp 2>&1 | FileCheck %s --check-prefix=EXTRACT_CROSS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/extract-word-crossing.cpp 2>&1 | FileCheck %s --check-prefix=EXTRACT_CROSS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/write-thread.cpp 2>&1 | FileCheck %s --check-prefix=WRITE_THREAD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/write-wide.cpp 2>&1 | FileCheck %s --check-prefix=WRITE_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-thread-runtime.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_THREAD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-thread-mixed.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_THREAD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/group-overlap.cpp 2>&1 | FileCheck %s --check-prefix=GROUP_OVERLAP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-tensix.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-thread.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_THREAD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-cross-end.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_END
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-outside-start.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_START
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/array-source-size.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY_SOURCE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-array-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_ARRAY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-shifted-array-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_ARRAY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-section-array-span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_ARRAY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/anchor-group-without-raw.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR_GROUP

// State: 40 + 184 reaches word 224; 40 + 0xffffffd8 would wrap to zero.
// Thread: 5 + 63 reaches word 68; 5 + 0xfffffffb would wrap to zero.
// thread-read-brisc.cpp builds without a TRISC thread define, as BRISC does.

// READ_ACCESS: error: static assertion failed: value-returning CFG reads require Access::MMIO
// SECTION: error: static assertion failed: section index out of range for this register
// READ_OFFSET: error: static assertion failed: CFG word offset crosses the selected bank
// ANCHOR_READ: error: static assertion failed: CFG word offset extends past its anchor field
// TARGET_STATE: error: static assertion failed: ThreadTarget applies only to thread CFG reads
// BRISC_TARGET: error: static assertion failed: BRISC thread-CFG reads must explicitly select ThreadTarget::T0, T1, or T2
// TRISC_ID: error: static assertion failed: COMPILE_FOR_TRISC must select TRISC0, TRISC1, or TRISC2
// EXTRACT_WIDE: error: static assertion failed: field wider than 32b cannot be extracted from a single value
// EXTRACT_CROSS: error: static assertion failed: field crosses a CFG word boundary
// WRITE_THREAD: error: static assertion failed: RISC writes target state CFG; use Access::TensixCfgUnit for thread CFG
// WRITE_WIDE: error: static assertion failed: field wider than 32b cannot be written through a single value
// GROUP_THREAD: error: static assertion failed: Access::MMIO cannot write thread CFG assignments
// GROUP_OVERLAP: error: static assertion failed: overlapping CFG field assignments in one physical word
// ARRAY_ACCESS: error: static assertion failed: array writes require Access::MMIO
// ARRAY_THREAD: error: static assertion failed: RISC writes target state CFG{{$}}
// ARRAY_END: error: static assertion failed: CFG array write crosses the end of the state bank
// ARRAY_START: error: static assertion failed: CFG array write starts outside the state bank
// ARRAY_SOURCE: error: static assertion failed: CFG word count exceeds source array
// ANCHOR_ARRAY: error: static assertion failed: CFG array write extends past its anchor field
// ANCHOR_GROUP: error: static assertion failed: whole-word CFG access requires a Field or a field group with a Raw anchor
// clang-format on

//--- fields.h
#pragma once

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Synthetic descriptors reach widths and bank boundaries that no named field occupies.
inline constexpr cfg::Field state_first {cfg::RegisterScope::State, 32, 0, 0, 0, 32, 1, 0};
inline constexpr cfg::Field state_outside {cfg::RegisterScope::State, 32, 224, 0, 0, 32, 1, 0};
inline constexpr cfg::Field thread_outside {cfg::RegisterScope::Thread, 16, 68, 0, 0, 16, 1, 0};
inline constexpr cfg::Field state_two_words {cfg::RegisterScope::State, 32, 64, 0, 0, 64, 1, 0};
inline constexpr cfg::Field state_word_crossing {cfg::RegisterScope::State, 32, 64, 0, 28, 8, 1, 0};

// The same 64-bit field spans words 64-66 in S0 (bit 16) and 67-68 in S1 (bit 0).
inline constexpr cfg::Field state_sectioned_wide {cfg::RegisterScope::State, 32, 64, 0, 16, 64, 2, 80};

// A field group without a Raw anchor cannot stand in for a Field.
class GroupWithoutRaw
{
public:
    static constexpr cfg::Field Value {cfg::RegisterScope::State, 32, 64, 0, 0, 32, 1, 0};
};

inline constexpr GroupWithoutRaw group_without_raw {};

//--- read-word-tensix.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>();
}

//--- read-field-tensix.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0>();
}

//--- read-section.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PrngSeed::Seed_Val, cfg::Sec::S1>();
}

//--- write-section.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::MMIO, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S1>(value);
}

//--- array-section.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::PrngSeed::Seed_Val, cfg::Sec::S1, 1>(values);
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

//--- anchor-read-span.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 4>();
}

//--- anchor-shifted-read-span.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 3>();
}

//--- anchor-section-read-span.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 2>();
}

//--- state-read-target.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PrngSeed::Seed_Val, cfg::Sec::S0, 0, cfg::ThreadTarget::T2>();
}

//--- thread-read-brisc.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0>();
}

//--- read-wide.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read<cfg::Access::MMIO, state_two_words, cfg::Sec::S0>();
}

//--- extract-wide.cpp
#include "fields.h"

std::uint32_t probe(std::uint32_t word)
{
    return cfg::extract<state_two_words, cfg::Sec::S0>(word);
}

//--- read-word-crossing.cpp
#include "fields.h"

std::uint32_t probe()
{
    return cfg::read<cfg::Access::MMIO, state_word_crossing, cfg::Sec::S0>();
}

//--- extract-word-crossing.cpp
#include "fields.h"

std::uint32_t probe(std::uint32_t word)
{
    return cfg::extract<state_word_crossing, cfg::Sec::S0>(word);
}

//--- write-thread.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0>(value);
}

//--- write-wide.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::MMIO, state_two_words, cfg::Sec::S0>(value);
}

//--- group-thread-runtime.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0>(value));
}

//--- group-thread-mixed.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(value), cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 1>());
}

//--- group-overlap.cpp
#include "fields.h"

void probe(std::uint32_t value)
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(), cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(value));
}

//--- array-tensix.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PrngSeed::Seed_Val, cfg::Sec::S0, 1>(values);
}

//--- array-thread.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, 1>(values);
}

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

//--- anchor-array-span.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 5>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 5>(values);
}

//--- anchor-shifted-array-span.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 4>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 4>(values);
}

//--- anchor-section-array-span.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 3>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 3>(values);
}

//--- anchor-group-without-raw.cpp
#include "fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    cfg::write<cfg::Access::MMIO, group_without_raw, cfg::Sec::S0, 1>(values);
}
