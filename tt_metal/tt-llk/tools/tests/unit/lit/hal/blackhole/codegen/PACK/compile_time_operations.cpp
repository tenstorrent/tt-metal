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
// PACR operands are (CfgContext, RowPadZero, DstAccessMode, AddrMod, AddrCntContext, ZeroWrite, ReadIntfSel,
// OvrdThreadId, Concat, CtxtCtrl, Flush, Last). The HAL leaves the operands the Blackhole ISA does not support at zero.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/pack.h"

namespace pack = hal::pack;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void transfer_defaults()
{
    pack::Packer0.transfer<pack::Transfer {}>();
}

// CHECK-LABEL: <transfer_defaults>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_every_option()
{
    pack::Packer0.transfer<pack::Transfer {
        .address_modifier = 3,
        .interfaces       = pack::ReadInterfaces::Row1And3,
        .datum_override   = pack::DatumOverride::Zero,
        .dest_access      = pack::DestAccess::Strided,
        .concatenation    = pack::Concatenation::Append,
        .boundary         = pack::TileBoundary::Last,
    }>();
}

// CHECK-LABEL: <transfer_every_option>:
// CHECK-NEXT: ttpacr 0,0,1,3,0,1,10,0,1,0,0,1{{$}}
// CHECK-NEXT: ret

// Each read-interface selection is its ReadIntfSel mask.
extern "C" __attribute__((noinline, used)) void transfer_read_interfaces()
{
    pack::Packer0.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row0}>();
    pack::Packer0.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row0To1}>();
    pack::Packer0.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row0To2}>();
    pack::Packer0.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row0And2}>();
    pack::Packer0.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row1And3}>();
}

// CHECK-LABEL: <transfer_read_interfaces>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,1,0,0,0,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,3,0,0,0,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,7,0,0,0,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,5,0,0,0,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,10,0,0,0,0,0{{$}}
// CHECK-NEXT: ret

// A tilize half-tile: Append rows on alternating interface pairs, then the row that closes the tile.
extern "C" __attribute__((noinline, used)) void transfer_tilize_rows()
{
    pack::Packer0.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row0And2, .concatenation = pack::Concatenation::Append}>();
    pack::Packer0
        .transfer<pack::Transfer {.address_modifier = 1, .interfaces = pack::ReadInterfaces::Row1And3, .concatenation = pack::Concatenation::Append}>();
    pack::Packer0.transfer<pack::Transfer {.address_modifier = 2, .interfaces = pack::ReadInterfaces::Row1And3, .boundary = pack::TileBoundary::Last}>();
}

// CHECK-LABEL: <transfer_tilize_rows>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,5,0,1,0,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,1,0,0,10,0,1,0,0,0{{$}}
// CHECK-NEXT: ttpacr 0,0,0,2,0,0,10,0,0,0,0,1{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

// Encoded transfers are plain words for MOP and replay programs.
extern "C" __attribute__((noinline, used)) void store_encoded_transfers(std::uint32_t* words)
{
    constexpr auto packer = pack::Packer0.operation();
    words[0]              = packer.transfer<pack::Transfer {.interfaces = pack::ReadInterfaces::Row0To1, .boundary = pack::TileBoundary::Last}>();
    words[1]              = packer.transfer({.address_modifier = 2, .dest_access = pack::DestAccess::Strided});
}

extern "C" __attribute__((noinline, used)) void reference_store_encoded_transfers(std::uint32_t* words)
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
    words[1] = TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        1 /*DstAccessMode*/,
        2 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        0 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        0 /*Last*/);
}

// The MOP pattern: encode a transfer, change one option, encode again.
extern "C" __attribute__((noinline, used)) void store_mop_transfers(std::uint32_t* words)
{
    constexpr auto packer = pack::Packer0.operation();
    pack::Transfer transfer {.interfaces = pack::ReadInterfaces::Row0, .datum_override = pack::DatumOverride::Zero};
    words[0]                  = packer.transfer(transfer);
    transfer.address_modifier = 1;
    transfer.boundary         = pack::TileBoundary::Last;
    words[1]                  = packer.transfer(transfer);
}

extern "C" __attribute__((noinline, used)) void reference_store_mop_transfers(std::uint32_t* words)
{
    words[0] = TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        0 /*DstAccessMode*/,
        0 /*AddrMode*/,
        0 /*AddrCntContext*/,
        1 /*ZeroWrite*/,
        0b0001 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        0 /*Last*/);
    words[1] = TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        0 /*DstAccessMode*/,
        1 /*AddrMode*/,
        0 /*AddrCntContext*/,
        1 /*ZeroWrite*/,
        0b0001 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        1 /*Last*/);
}

// A template transfer issues the same immediate as production's TTI_PACR.
extern "C" __attribute__((noinline, used)) void issue_template_transfer()
{
    pack::Packer0.transfer<pack::Transfer {.address_modifier = 1, .interfaces = pack::ReadInterfaces::Row0To2, .dest_access = pack::DestAccess::Strided}>();
}

extern "C" __attribute__((noinline, used)) void reference_issue_template_transfer()
{
    TTI_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        1 /*DstAccessMode*/,
        1 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        0b0111 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        0 /*Last*/);
}
