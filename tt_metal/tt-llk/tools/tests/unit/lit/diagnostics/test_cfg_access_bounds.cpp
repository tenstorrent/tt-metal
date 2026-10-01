// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: blackhole-cfg, sfpi
// DEFINE: %{check} = %{sfpi_cxx} %{cfg_flags} -DCOMPILE_FOR_TRISC=0 -fsyntax-only -fmax-errors=3 -I %S/../Inputs
// RUN: split-file %s %t
//
// RUN: not %{check} %t/array_cross_end.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY-END
// RUN: not %{check} %t/array_outside_start.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY-START
// RUN: not %{check} %t/array_source_size.cpp 2>&1 | FileCheck %s --check-prefix=ARRAY-SOURCE
// RUN: not %{check} %t/state_read_end.cpp 2>&1 | FileCheck %s --check-prefix=READ-OFFSET
// RUN: not %{check} %t/state_read_wrap.cpp 2>&1 | FileCheck %s --check-prefix=READ-OFFSET
// RUN: not %{check} %t/state_read_outside.cpp 2>&1 | FileCheck %s --check-prefix=READ-OFFSET
// RUN: not %{check} %t/thread_read_end.cpp 2>&1 | FileCheck %s --check-prefix=READ-OFFSET
// RUN: not %{check} %t/thread_read_wrap.cpp 2>&1 | FileCheck %s --check-prefix=READ-OFFSET
// RUN: not %{check} %t/thread_read_outside.cpp 2>&1 | FileCheck %s --check-prefix=READ-OFFSET
// RUN: not %{check} %t/gpr_read_outside.cpp 2>&1 | FileCheck %s --check-prefix=GPR-READ
// RUN: not %{check} %t/gpr_direct_state_outside_Bits32.cpp 2>&1 | FileCheck %s --check-prefix=GPR-ADDR
// RUN: not %{check} %t/gpr_direct_state_outside_Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR-ADDR
// RUN: not %{check} %t/gpr_direct_state_crossing_Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR-SPAN
// RUN: not %{check} %t/gpr_grouped_state_outside_Bits32.cpp 2>&1 | FileCheck %s --check-prefix=GPR-ADDR
// RUN: not %{check} %t/gpr_grouped_state_outside_Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR-ADDR
// RUN: not %{check} %t/gpr_grouped_state_crossing_Bits128.cpp 2>&1 | FileCheck %s --check-prefix=GPR-SPAN
// RUN: not %{check} %t/gpr_operand_outside.cpp 2>&1 | FileCheck %s --check-prefix=GPR-ADDR
// RUN: not %{check} %t/gpr_operand_span.cpp 2>&1 | FileCheck %s --check-prefix=GPR-SPAN
// RUN: not %{check} %t/anchor_array_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-ARRAY
// RUN: not %{check} %t/anchor_read_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-READ
// RUN: not %{check} %t/anchor_gpr_direct_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-GPR
// RUN: not %{check} %t/anchor_gpr_grouped_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-GPR
// RUN: not %{check} %t/anchor_group_without_raw.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-GROUP
// RUN: not %{check} %t/anchor_shifted_array_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-ARRAY
// RUN: not %{check} %t/anchor_shifted_read_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-READ
// RUN: not %{check} %t/anchor_section_array_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-ARRAY
// RUN: not %{check} %t/anchor_section_read_span.cpp 2>&1 | FileCheck %s --check-prefix=ANCHOR-READ

// Match an actual compiler error, not the echoed static_assert source text.
// ARRAY-END: error: static assertion failed: CFG array write crosses the end of the state bank
// ARRAY-START: error: static assertion failed: CFG array write starts outside the state bank
// ARRAY-SOURCE: error: static assertion failed: CFG word count exceeds source array
// READ-OFFSET: error: static assertion failed: CFG word offset crosses the selected bank
// GPR-READ: error: static assertion failed: CFG read source lies outside the state bank
// GPR-ADDR: error: static assertion failed: CFG write destination lies outside its register scope
// GPR-SPAN: error: static assertion failed: GPR write crosses the end of its CFG bank
// ANCHOR-ARRAY: error: static assertion failed: CFG array write extends past its anchor field
// ANCHOR-READ: error: static assertion failed: CFG word offset extends past its anchor field
// ANCHOR-GPR: error: static assertion failed: GPR write extends past its anchor field
// ANCHOR-GROUP: error: static assertion failed: whole-word CFG access requires a Field or a field group with a Raw anchor

// State: 40 + 184 reaches word 224; 40 + 0xffffffd8 would wrap to zero.
// Thread: 5 + 63 reaches word 68; 5 + 0xfffffffb would wrap to zero.

//--- array_cross_end.cpp
#include <cstdint>

#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 3>& values)
{
    write<Access::MMIO, ChickenBits::sfpu_scbd_disable, Sec::S0, 3>(values);
}

//--- array_outside_start.cpp
#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    write<Access::MMIO, state_outside, Sec::S0, 1>(values);
}

//--- array_source_size.cpp
#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    write<Access::MMIO, state_first, Sec::S0, 2>(values);
}

//--- state_read_end.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, Sec::S0, 184>();
}

//--- state_read_wrap.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, Sec::S0, 0xffffffd8u>();
}

//--- state_read_outside.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, state_outside, Sec::S0>();
}

//--- thread_read_end.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, SrcASet::Base, Sec::S0, 63>();
}

//--- thread_read_wrap.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, SrcASet::Base, Sec::S0, 0xfffffffbu>();
}

//--- thread_read_outside.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, thread_outside, Sec::S0>();
}

//--- gpr_read_outside.cpp
#include "cfg_test_fields.h"

void probe()
{
    read<Access::TensixCfgUnit, state_outside, Sec::S0>(hal::gpr<4>());
}

//--- gpr_direct_state_outside_Bits32.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit, state_outside, Sec::S0, GprTransferSize::Bits32>(hal::gpr<4>());
}

//--- gpr_direct_state_outside_Bits128.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit, state_outside, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- gpr_direct_state_crossing_Bits128.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit, state_crossing, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- gpr_grouped_state_outside_Bits32.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit>(from_gpr<state_outside, Sec::S0, GprTransferSize::Bits32>(hal::gpr<4>()));
}

//--- gpr_grouped_state_outside_Bits128.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit>(from_gpr<state_outside, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- gpr_grouped_state_crossing_Bits128.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit>(from_gpr<state_crossing, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- gpr_operand_outside.cpp
#include "cfg_test_fields.h"

void probe()
{
    auto operand = from_gpr<state_outside, Sec::S0>(hal::gpr<4>());
}

//--- gpr_operand_span.cpp
#include "cfg_test_fields.h"

void probe()
{
    auto operand = from_gpr<state_crossing, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- anchor_array_span.cpp
#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 5>& values)
{
    write<Access::MMIO, Thcon[Reg0].TileDescriptor, Sec::S0, 5>(values);
}

//--- anchor_read_span.cpp
#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, Thcon[Reg0].TileDescriptor, Sec::S0, 4>();
}

//--- anchor_gpr_direct_span.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit, state_two_words, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>());
}

//--- anchor_gpr_grouped_span.cpp
#include "cfg_test_fields.h"

void probe()
{
    write<Access::TensixCfgUnit>(from_gpr<state_two_words, Sec::S0, GprTransferSize::Bits128>(hal::gpr<4>()));
}

//--- anchor_group_without_raw.cpp
#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 1>& values)
{
    write<Access::MMIO, group_without_raw, Sec::S0, 1>(values);
}

//--- anchor_shifted_array_span.cpp
#include <cstdint>

#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 4>& values)
{
    write<Access::MMIO, state_sectioned_wide, Sec::S0, 4>(values);
}

//--- anchor_shifted_read_span.cpp
#include <cstdint>

#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, state_sectioned_wide, Sec::S0, 3>();
}

//--- anchor_section_array_span.cpp
#include <cstdint>

#include "cfg_test_fields.h"

void probe(const std::array<std::uint32_t, 3>& values)
{
    write<Access::MMIO, state_sectioned_wide, Sec::S1, 3>(values);
}

//--- anchor_section_read_span.cpp
#include <cstdint>

#include "cfg_test_fields.h"

std::uint32_t probe()
{
    return read_word<Access::MMIO, state_sectioned_wide, Sec::S1, 2>();
}
