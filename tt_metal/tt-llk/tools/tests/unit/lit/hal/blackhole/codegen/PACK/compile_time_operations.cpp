// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump

// Compare each HAL function in codegen.cpp with its adjacent reference_
// function. Disable identical-code folding so both bodies are emitted
// independently.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/pack.h"

namespace pack = hal::pack;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void transfer_defaults()
{
    pack::run<pack::DataTransfer {}>();
}

// CHECK-LABEL: <transfer_defaults>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_every_field()
{
    pack::run<pack::DataTransfer {
        .address_modifier      = 3,
        .context               = pack::ContextControl::HardwareCounterNoAdvance,
        .configuration_context = 3,
        .counter_context       = 2,
        .override_thread_id    = true,
        .interfaces            = 0xf,
        .datum_override        = pack::DatumOverride::Zero,
        .dest_access           = pack::DestAccess::Strided,
        .padding               = pack::RowPadding::FinalOnly,
        .alignment             = pack::PaddingAlignment::To16Datums,
        .concatenation         = pack::Concatenation::Append,
        .boundary              = pack::TileBoundary::Last,
    }>();
}

// CHECK-LABEL: <transfer_every_field>:
// CHECK-NEXT: ttpacr 3,7,1,3,2,1,15,1,1,3,0,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_context_controls()
{
    pack::run<pack::DataTransfer {.context = pack::ContextControl::HardwareCounter, .interfaces = 0b0101}>();
    pack::run<pack::DataTransfer {.context = pack::ContextControl::HardwareCounterReset}>();
    pack::run<pack::DataTransfer {.padding = pack::RowPadding::AllTransfers, .boundary = pack::TileBoundary::Last}>();
    pack::run<pack::DataTransfer {.padding = pack::RowPadding::NonConcatenated, .alignment = pack::PaddingAlignment::To16Datums}>();
}

// CHECK-LABEL: <transfer_context_controls>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,5,0,0,1,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,2,0,0{{$}}
// CHECK-NEXT: ttpacr 0,1,0,0,0,0,0,0,0,0,0,1{{$}}
// CHECK-NEXT: ttpacr 0,6,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ret

// The value is staged in two 16-bit halves before the write consumes it.
extern "C" __attribute__((noinline, used)) void register_write()
{
    pack::run<pack::RegisterWriteValue {.value = 0x1234}>();
    pack::run<pack::RegisterWriteValue {.high_half = true, .value = 0xabcd}>();
    pack::run<pack::RegisterWrite {.address_slot = 3, .stream_id = 63}>();
}

// The load flavor's value (bits 18:3) and half select (bit 2) overlap the
// write flavor's StreamId..Unused fields; objdump decodes them as those.
// CHECK-LABEL: <register_write>:
// CHECK-NEXT: ttpacrsetreg 0,1,9,0,1,40,0,0{{$}}
// CHECK-NEXT: ttpacrsetreg 0,1,85,3,2,27,0,0{{$}}
// CHECK-NEXT: ttpacrsetreg 1,0,0,0,3,63,1,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void edge_window()
{
    pack::run<pack::EdgeWindow {.x_start = 1, .x_end = 2, .y_start = 3, .y_end = 15}>();
}

// CHECK-LABEL: <edge_window>:
// CHECK-NEXT: ttsetpkedgof 15,3,2,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void flush_and_clear()
{
    pack::flush_write_aligners();
    pack::clear_exponent_histogram();
}

// CHECK-LABEL: <flush_and_clear>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,0,1,0{{$}}
// CHECK-NEXT: ttclrexphist
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    constexpr std::uint32_t transfer = pack::DataTransfer {.interfaces = 0b0011, .boundary = pack::TileBoundary::Last}.get_operation();
    constexpr std::uint32_t write    = pack::RegisterWrite {.address_slot = 1}.get_operation();
    constexpr std::uint32_t value    = pack::RegisterWriteValue {.value = 0xffff}.get_operation();
    constexpr std::uint32_t window   = pack::EdgeWindow {.y_end = 1}.get_operation();
    words[0]                         = transfer;
    words[1]                         = write;
    words[2]                         = value;
    words[3]                         = window;
    words[4]                         = pack::flush_write_aligners_operation();
    words[5]                         = pack::clear_exponent_histogram_operation();
}

// Value 0xffff sits in bits 18:3, which the write-flavor macro splits into
// Unused = 0x7f, DisableStall = 3, AddrSel = 3 and StreamId = 0x3e.
extern "C" __attribute__((noinline, used)) void reference_store_encoded_operations(std::uint32_t* words)
{
    words[0] = TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        0 /*DstAccessMode*/,
        0 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        0b0011 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        1 /*Last*/);
    words[1] = TT_OP_PACR_SETREG(1 /*Push*/, 0 /*ModeSel*/, 0 /*Unused*/, 0 /*DisableStall*/, 1 /*AddrSel*/, 0 /*StreamId*/, 1 /*Flush*/, 0 /*Last*/);
    words[2] = TT_OP_PACR_SETREG(0 /*Push*/, 1 /*ModeSel*/, 0x7f /*Unused*/, 3 /*DisableStall*/, 3 /*AddrSel*/, 0x3e /*StreamId*/, 0 /*Flush*/, 0 /*Last*/);
    words[3] = TT_OP_SETPKEDGOF(1 /*y_end*/, 0 /*y_start*/, 0 /*x_end*/, 0 /*x_start*/);
    words[4] = TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        0 /*DstAccessMode*/,
        0 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        0 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        1 /*Flush*/,
        0 /*Last*/);
    words[5] = TT_OP_CLREXPHIST;
}
