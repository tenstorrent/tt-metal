// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_unpack_thread}
// RUN: %split-file %s %t
// DEFINE: %{production_warnings} = -Wall -Werror -Wno-error=deprecated-declarations -Wunused-parameter -Wfloat-equal -Wpointer-arith -Wnull-dereference -Wredundant-decls -Wuninitialized -Wmaybe-uninitialized
// RUN: %{diagnose} %t/valid.cpp
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} %{production_warnings} -c %t/warnings.cpp -o %t/warnings.o
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} %{production_warnings} -DENABLE_LLK_ASSERT -c %t/warnings.cpp -o %t/warnings.assert.o
// RUN: not %{diagnose} %t/invalid-section.cpp 2>&1 | FileCheck %s --check-prefix=SECTION
// RUN: not %{diagnose} %t/invalid-value.cpp 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: not %{diagnose} %t/packer.cpp 2>&1 | FileCheck %s --check-prefix=PACKER
// RUN: not %{diagnose} %t/packer-addr-ctrl.cpp 2>&1 | FileCheck %s --check-prefix=PACKER_ADDR_CTRL
// RUN: not %{diagnose} %t/packer-addr-base.cpp 2>&1 | FileCheck %s --check-prefix=PACKER_ADDR_BASE
// RUN: not %{diagnose} %t/unpacker.cpp 2>&1 | FileCheck %s --check-prefix=UNPACKER
// RUN: not %{diagnose} %t/unpacker-addr-ctrl.cpp 2>&1 | FileCheck %s --check-prefix=UNPACKER_REG
// RUN: not %{diagnose} %t/unpacker-addr-base.cpp 2>&1 | FileCheck %s --check-prefix=UNPACKER_REG
// RUN: not %{diagnose} %t/context-unpacker-0.cpp 2>&1 | FileCheck %s --check-prefix=CONTEXT
// RUN: not %{diagnose} %t/context-unpacker-1.cpp 2>&1 | FileCheck %s --check-prefix=CONTEXT
// RUN: not %{diagnose} %t/row-set-mapping.cpp 2>&1 | FileCheck %s --check-prefix=ROW_MAPPING
// RUN: not %{diagnose} %t/row-set-set.cpp 2>&1 | FileCheck %s --check-prefix=ROW_SET
// RUN: not %{diagnose} %t/face-set-mapping.cpp 2>&1 | FileCheck %s --check-prefix=FACE_MAPPING
// RUN: not %{diagnose} %t/face-set-set.cpp 2>&1 | FileCheck %s --check-prefix=FACE_SET
// RUN: not %{diagnose} %t/perf-counter.cpp 2>&1 | FileCheck %s --check-prefix=PERF_COUNTER
// RUN: not %{diagnose} %t/perf-counter-integral.cpp 2>&1 | FileCheck %s --check-prefix=INTEGRAL_INDEX
// RUN: not %{diagnose} %t/unpacker-integral.cpp 2>&1 | FileCheck %s --check-prefix=INTEGRAL_INDEX
// clang-format on

// SECTION: error: static assertion failed: section index out of range for this register
// VALUE: error: static assertion failed: value exceeds field width
// PACKER: error: static assertion failed: Blackhole exposes address configuration only for packer 0
// PACKER_ADDR_CTRL: error: static assertion failed: packer address-control register index out of range
// PACKER_ADDR_BASE: error: static assertion failed: packer address-base register index out of range
// UNPACKER: error: static assertion failed: unpacker index out of range
// UNPACKER_REG: error: static assertion failed: unpacker register index out of range
// CONTEXT: error: static assertion failed: unpacker context index out of range
// ROW_MAPPING: error: static assertion failed: tile row-set mapping index out of range
// ROW_SET: error: static assertion failed: tile row-set set index out of range
// FACE_MAPPING: error: static assertion failed: tile face-set mapping index out of range
// FACE_SET: error: static assertion failed: tile face-set set index out of range
// PERF_COUNTER: error: static assertion failed: performance counter index out of range
// INTEGRAL_INDEX: error: call to non-'constexpr' function {{.*}}invalid_index

//--- fields.h
#pragma once

#include <cstdint>
#include <type_traits>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

template <std::uint32_t I>
using Index = std::integral_constant<std::uint32_t, I>;

//--- valid.cpp
#include "fields.h"

static_assert(static_cast<std::uint32_t>(cfg::Access::MMIO) == 0);
static_assert(static_cast<std::uint32_t>(cfg::Access::TensixCfgUnit) == 1);
static_assert(static_cast<std::uint32_t>(cfg::Access::TensixScalarUnit) == 2);
static_assert(static_cast<std::uint32_t>(cfg::RegisterScope::Thread) == 0);
static_assert(static_cast<std::uint32_t>(cfg::RegisterScope::State) == 1);
static_assert(static_cast<std::uint32_t>(cfg::Sec::S0) == 0);
static_assert(static_cast<std::uint32_t>(cfg::Sec::S7) == 7);
static_assert(static_cast<std::uint32_t>(cfg::ThreadTarget::Current) == 0);
static_assert(static_cast<std::uint32_t>(cfg::ThreadTarget::T2) == 3);
static_assert(static_cast<std::uint32_t>(cfg::GprTransferSize::Bits32) == 0);
static_assert(static_cast<std::uint32_t>(cfg::GprTransferSize::Bits128) == 1);
static_assert(static_cast<std::uint32_t>(cfg::WrcfgCompletion::Wait) == 0);
static_assert(static_cast<std::uint32_t>(cfg::WrcfgCompletion::Deferred) == 1);

// Field arithmetic: state/thread words, repeated sections, and narrow/full-width masks.
constexpr cfg::Field state_field {cfg::RegisterScope::State, 32, 10, 1, 3, 5, 3, 40};
static_assert(state_field.abs0() == 355);
static_assert(state_field.addr32(cfg::Sec::S0) == 11);
static_assert(state_field.addr32(cfg::Sec::S1) == 12);
static_assert(state_field.shamt(cfg::Sec::S1) == 11);
static_assert(state_field.mask(cfg::Sec::S1) == 0xf800u);

constexpr cfg::Field full_word {cfg::RegisterScope::State, 32, 20, 0, 0, 32, 1, 0};
static_assert(full_word.mask(cfg::Sec::S0) == 0xffffffffu);

constexpr cfg::Field thread_field {cfg::RegisterScope::Thread, 16, 7, 0, 8, 4, 2, 16};
static_assert(thread_field.abs0() == 120);
static_assert(thread_field.addr32(cfg::Sec::S1) == 8);
static_assert(thread_field.shamt(cfg::Sec::S1) == 8);
static_assert(thread_field.mask(cfg::Sec::S1) == 0x0f00u);

// Every typed THCON selector overload resolves a representative field.
static_assert(cfg::Thcon[cfg::Reg0].TileDescriptor.Raw.addr32(cfg::Sec::S0) == 64);
static_assert(cfg::Thcon[cfg::Reg1].L1_Dest_addr.addr32(cfg::Sec::S0) == 69);
static_assert(cfg::Thcon[cfg::Reg2].Out_data_format.addr32(cfg::Sec::S0) == 72);
static_assert(cfg::Thcon[cfg::Reg3].Base_address.addr32(cfg::Sec::S0) == 76);
static_assert(cfg::Thcon[cfg::Reg4].Base_cntx4_address.addr32(cfg::Sec::S0) == 80);
static_assert(cfg::Thcon[cfg::Reg5].Dest_cntx0_address.addr32(cfg::Sec::S0) == 84);
static_assert(cfg::Thcon[cfg::Reg6].Source_address.addr32(cfg::Sec::S0) == 88);
static_assert(cfg::Thcon[cfg::Reg7].Offset_address.addr32(cfg::Sec::S0) == 92);
static_assert(cfg::Thcon[cfg::Reg8].L1_Dest_addr.addr32(cfg::Sec::S0) == 97);
static_assert(cfg::Thcon[cfg::Reg9].Pack_0_2_limit_address.addr32(cfg::Sec::S0) == 100);
static_assert(cfg::Thcon[cfg::Reg10].Unpack_limit_address.addr32(cfg::Sec::S0) == 104);
static_assert(cfg::Thcon[cfg::Reg11].Metadata_l1_addr.addr32(cfg::Sec::S0) == 108);

// Selector-only descriptor entries.
static_assert(cfg::DisableImpliedFmt[cfg::SrcA].addr32(cfg::Sec::S0) == 2);
static_assert(cfg::DisableImpliedFmt[cfg::SrcB].addr32(cfg::Sec::S0) == 3);
static_assert(cfg::AddrMod[cfg::SrcA].Incr.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::AddrMod[cfg::SrcB].Incr.shamt(cfg::Sec::S0) == 8);
static_assert(cfg::AddrMod[cfg::Src][cfg::Y].Incr.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::AddrMod[cfg::Src][cfg::Z].Incr.shamt(cfg::Sec::S0) == 12);
static_assert(cfg::AddrMod[cfg::Dest].Incr.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::AddrMod[cfg::Dest][cfg::Y].Incr.shamt(cfg::Sec::S0) == 6);
static_assert(cfg::AddrMod[cfg::Dest][cfg::Z].Incr.shamt(cfg::Sec::S0) == 14);
static_assert(cfg::AddrMod[cfg::Fidelity].Incr.shamt(cfg::Sec::S0) == 13);
static_assert(cfg::AddrMod[cfg::Bias].Incr.shamt(cfg::Sec::S0) == 0);

// Expected CFG word addresses and bit shifts are independent of the descriptor builders.
// Counts plus visited-index masks require each entry exactly once, in any order.
constexpr std::uint32_t packer_control_words[]      = {12, 14};
constexpr std::uint32_t packer_base_words[]         = {16, 17};
constexpr std::uint32_t unpacker_control_words[][2] = {{44, 56}, {46, 58}};
constexpr std::uint32_t unpacker_base_words[][2]    = {{48, 49}, {60, 61}};
constexpr std::uint32_t unpacker_context_words[][8] = {{76, 77, 78, 79, 80, 81, 82, 83}, {124, 125}};
constexpr std::uint32_t row_mapping_words[]         = {20, 21, 22, 23};
constexpr std::uint32_t face_mapping_words[]        = {36, 37, 38, 39};
constexpr std::uint32_t mapping_shifts[]            = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};

constexpr bool packer_descriptors_are_complete()
{
    constexpr auto packer       = cfg::Packer[Index<0> {}];
    std::uint32_t control_count = 0;
    std::uint32_t controls_seen = 0;
    packer.AddrCtrl.forEach(
        [&](auto reg)
        {
            ++control_count;
            controls_seen |= 1u << reg;
            constexpr auto selected = cfg::Packer[Index<0> {}].AddrCtrl[decltype(reg) {}];
            static_assert(selected.Xstride.addr32(cfg::Sec::S0) == packer_control_words[reg], "packer control word");
            static_assert(selected.Xstride.shamt(cfg::Sec::S0) == 0, "packer X stride shift");
            static_assert(selected.Zstride.addr32(cfg::Sec::S0) == packer_control_words[reg] + 1, "packer Z stride word");
        });

    std::uint32_t base_count = 0;
    std::uint32_t bases_seen = 0;
    packer.AddrBase.forEach(
        [&](auto reg)
        {
            ++base_count;
            bases_seen |= 1u << reg;
            constexpr const auto& field = cfg::Packer[Index<0> {}].AddrBase[decltype(reg) {}];
            static_assert(field.addr32(cfg::Sec::S0) == packer_base_words[reg], "packer base word");
            static_assert(field.shamt(cfg::Sec::S0) == 0, "packer base shift");
        });

    return control_count == 2 && controls_seen == 0x3u && base_count == 2 && bases_seen == 0x3u &&
           packer.AddrCtrl[cfg::PackerReg::Reg0].Zstride.addr32(cfg::Sec::S0) == 13 &&
           packer.AddrCtrl[cfg::PackerReg::Reg1].Zstride.addr32(cfg::Sec::S0) == 15 && packer.AddrBase[cfg::PackerReg::Reg0].addr32(cfg::Sec::S0) == 16 &&
           packer.AddrBase[cfg::PackerReg::Reg1].addr32(cfg::Sec::S0) == 17;
}

static_assert(packer_descriptors_are_complete());

constexpr bool unpacker_descriptors_are_complete()
{
    std::uint32_t unpacker_count = 0;
    std::uint32_t register_count = 0;
    std::uint32_t context_count  = 0;
    std::uint32_t unpackers_seen = 0;
    std::uint32_t controls_seen  = 0;
    std::uint32_t bases_seen     = 0;
    std::uint32_t contexts_seen  = 0;

    cfg::Unpacker.forEach(
        [&](auto unpacker)
        {
            ++unpacker_count;
            unpackers_seen |= 1u << unpacker;
            constexpr auto selected = cfg::Unpacker[decltype(unpacker) {}];
            selected.AddrCtrl.forEach(
                [&](auto reg)
                {
                    ++register_count;
                    controls_seen |= 1u << (2 * unpacker + reg);
                    constexpr const auto& field = cfg::Unpacker[decltype(unpacker) {}].AddrCtrl[decltype(reg) {}].Xstride;
                    static_assert(field.addr32(cfg::Sec::S0) == unpacker_control_words[unpacker][reg], "unpacker control word");
                    static_assert(field.shamt(cfg::Sec::S0) == 0, "unpacker X stride shift");
                });
            selected.AddrBase.forEach(
                [&](auto reg)
                {
                    ++register_count;
                    bases_seen |= 1u << (2 * unpacker + reg);
                    constexpr const auto& field = cfg::Unpacker[decltype(unpacker) {}].AddrBase[decltype(reg) {}];
                    static_assert(field.addr32(cfg::Sec::S0) == unpacker_base_words[unpacker][reg], "unpacker base word");
                    static_assert(field.shamt(cfg::Sec::S0) == 0, "unpacker base shift");
                });
            selected.Cntx.forEach(
                [&](auto context)
                {
                    ++context_count;
                    // Unpacker 0 has eight contexts; unpacker 1 has two.
                    contexts_seen |= 1u << (8 * unpacker + context);
                    constexpr const auto& field = cfg::Unpacker[decltype(unpacker) {}].Cntx[decltype(context) {}].Base;
                    static_assert(field.addr32(cfg::Sec::S0) == unpacker_context_words[unpacker][context], "unpacker context word");
                    static_assert(field.shamt(cfg::Sec::S0) == 0, "unpacker context shift");
                });
        });

    return unpacker_count == 2 && register_count == 8 && context_count == 10 && unpackers_seen == 0x3u && controls_seen == 0xfu && bases_seen == 0xfu &&
           contexts_seen == 0x3ffu && cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg0].Xstride.addr32(cfg::Sec::S0) == 44 &&
           cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1].Xstride.addr32(cfg::Sec::S0) == 56 &&
           cfg::Unpacker[1].AddrBase[cfg::UnpackerReg::Reg0].addr32(cfg::Sec::S0) == 60 &&
           cfg::Unpacker[1].AddrBase[cfg::UnpackerReg::Reg1].addr32(cfg::Sec::S0) == 61 &&
           cfg::Unpacker[0].BlobsYStart[cfg::BlobContext::Context01].addr32(cfg::Sec::S0) == 51 &&
           cfg::Unpacker[0].BlobsYStart[cfg::BlobContext::Context23].addr32(cfg::Sec::S0) == 52;
}

static_assert(unpacker_descriptors_are_complete());

constexpr bool mapping_descriptors_are_complete()
{
    std::uint32_t row_fields  = 0;
    std::uint32_t face_fields = 0;
    std::uint64_t rows_seen   = 0;
    std::uint64_t faces_seen  = 0;

    cfg::TileRowSetMapping.forEach(
        [&](auto mapping)
        {
            cfg::TileRowSetMapping[mapping].forEach(
                [&](auto set)
                {
                    ++row_fields;
                    rows_seen |= std::uint64_t {1} << (16 * mapping + set);
                    constexpr const auto& field = cfg::TileRowSetMapping[decltype(mapping) {}][decltype(set) {}];
                    static_assert(field.addr32(cfg::Sec::S0) == row_mapping_words[mapping], "tile row-set mapping word");
                    static_assert(field.shamt(cfg::Sec::S0) == mapping_shifts[set], "tile row-set mapping shift");
                });
        });
    cfg::TileFaceSetMapping.forEach(
        [&](auto mapping)
        {
            cfg::TileFaceSetMapping[mapping].forEach(
                [&](auto set)
                {
                    ++face_fields;
                    faces_seen |= std::uint64_t {1} << (16 * mapping + set);
                    constexpr const auto& field = cfg::TileFaceSetMapping[decltype(mapping) {}][decltype(set) {}];
                    static_assert(field.addr32(cfg::Sec::S0) == face_mapping_words[mapping], "tile face-set mapping word");
                    static_assert(field.shamt(cfg::Sec::S0) == mapping_shifts[set], "tile face-set mapping shift");
                });
        });

    return row_fields == 64 && face_fields == 64 && rows_seen == 0xffffffffffffffffull && faces_seen == 0xffffffffffffffffull;
}

static_assert(mapping_descriptors_are_complete());

static_assert(cfg::PerfCntCmd[0].Start.shamt(cfg::Sec::S0) == 0);
static_assert(cfg::PerfCntCmd[1].Start.shamt(cfg::Sec::S0) == 2);
static_assert(cfg::PerfCntCmd[2].Stop.shamt(cfg::Sec::S0) == 5);
static_assert(cfg::PerfCntCmd[3].Stop.shamt(cfg::Sec::S0) == 7);
// Both ends of the public integral-index overloads remain valid.
static_assert(cfg::Unpacker[0u].Cntx[0u].Base.addr32(cfg::Sec::S0) == 76);
static_assert(cfg::Unpacker[1u].Cntx[1u].Base.addr32(cfg::Sec::S0) == 125);
constexpr auto first_perf_counter = cfg::PerfCntCmdEntry::make<0>();
constexpr auto last_perf_counter  = cfg::PerfCntCmdEntry::make<3>();
static_assert(first_perf_counter.Start.shamt(cfg::Sec::S0) == 0);
static_assert(last_perf_counter.Stop.shamt(cfg::Sec::S0) == 7);

// Compile-time builders and their public descriptor results.
constexpr auto common_gpr = hal::gpr<5>();
static_assert(decltype(common_gpr)::index == 5);

constexpr auto constant_assignment = cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>();
static_assert(decltype(constant_assignment)::scope == cfg::RegisterScope::State);
static_assert(decltype(constant_assignment)::addr == 1);
static_assert(decltype(constant_assignment)::shift == 29);
static_assert(decltype(constant_assignment)::mask == 0x20000000u);
static_assert(decltype(constant_assignment)::value == 1);

constexpr auto gpr_write =
    cfg::from_gpr<cfg::Thcon[cfg::Reg0].TileDescriptor.Raw, cfg::Sec::S1, cfg::GprTransferSize::Bits128, cfg::WrcfgCompletion::Deferred>(hal::gpr<16>());
static_assert(decltype(gpr_write)::scope == cfg::RegisterScope::State);
static_assert(decltype(gpr_write)::addr == 112);
static_assert(decltype(gpr_write)::words == 4);
static_assert(decltype(gpr_write.source)::index == 16);

constexpr auto common_gpr_write = cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(common_gpr);
static_assert(decltype(common_gpr_write)::addr == 76);
static_assert(decltype(common_gpr_write)::words == 1);

//--- warnings.cpp
#include "fields.h"

// Positive integration smoke test: instantiate public read/write paths under
// production warning flags, both with and without runtime assertions.
void write_fields(std::uint32_t value)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AddrMod[cfg::SrcA].Incr, cfg::Sec::S0>(value);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(value),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>(),
        cfg::from_gpr<cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(hal::gpr<4>()));
    cfg::write<cfg::Access::MMIO, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(value);
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(value),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>());
}

std::uint32_t read_field()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0>();
}

//--- invalid-section.cpp
#include "fields.h"

constexpr auto invalid_section = cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S1, 1>();

//--- invalid-value.cpp
#include "fields.h"

constexpr auto invalid_value = cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 2>();

//--- packer.cpp
#include "fields.h"

constexpr auto invalid_packer = cfg::Packer[Index<1> {}];

//--- packer-addr-ctrl.cpp
#include "fields.h"

constexpr auto invalid_register = cfg::Packer[Index<0> {}].AddrCtrl[Index<2> {}];

//--- packer-addr-base.cpp
#include "fields.h"

constexpr const cfg::Field& invalid_register = cfg::Packer[Index<0> {}].AddrBase[Index<2> {}];

//--- unpacker.cpp
#include "fields.h"

constexpr auto invalid_unpacker = cfg::Unpacker[Index<2> {}];

//--- unpacker-addr-ctrl.cpp
#include "fields.h"

constexpr auto invalid_register = cfg::Unpacker[Index<0> {}].AddrCtrl[Index<2> {}];

//--- unpacker-addr-base.cpp
#include "fields.h"

constexpr const cfg::Field& invalid_register = cfg::Unpacker[Index<1> {}].AddrBase[Index<2> {}];

//--- context-unpacker-0.cpp
#include "fields.h"

constexpr auto invalid_context = cfg::Unpacker[Index<0> {}].Cntx[Index<8> {}];

//--- context-unpacker-1.cpp
#include "fields.h"

constexpr auto invalid_context = cfg::Unpacker[Index<1> {}].Cntx[Index<2> {}];

//--- row-set-mapping.cpp
#include "fields.h"

constexpr auto invalid_mapping = cfg::TileRowSetMapping[Index<4> {}];

//--- row-set-set.cpp
#include "fields.h"

constexpr const cfg::Field& invalid_set = cfg::TileRowSetMapping[Index<3> {}][Index<16> {}];

//--- face-set-mapping.cpp
#include "fields.h"

constexpr auto invalid_mapping = cfg::TileFaceSetMapping[Index<4> {}];

//--- face-set-set.cpp
#include "fields.h"

constexpr const cfg::Field& invalid_set = cfg::TileFaceSetMapping[Index<3> {}][Index<16> {}];

//--- perf-counter.cpp
#include "fields.h"

constexpr auto invalid_counter = cfg::PerfCntCmd[Index<4> {}];

//--- perf-counter-integral.cpp
#include "fields.h"

constexpr auto invalid_counter = cfg::PerfCntCmd[4u];

//--- unpacker-integral.cpp
#include "fields.h"

constexpr auto invalid_unpacker = cfg::Unpacker[2u];
