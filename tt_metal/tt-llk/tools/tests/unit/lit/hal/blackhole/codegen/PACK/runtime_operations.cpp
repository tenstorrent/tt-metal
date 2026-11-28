// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// codegen.cpp is compiled without LLK_ASSERT so runtime paths contain only the
// issued instruction; asserts.cpp checks the runtime operand asserts.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/pack.h"

namespace pack = hal::pack;

//--- codegen.cpp
#include "common.h"

// Runtime options are encoded in a register and pushed through the instruction buffer.

extern "C" __attribute__((noinline, used)) void run_runtime_interfaces(const pack::ReadInterfaces interfaces)
{
    pack::Packer0.transfer({.interfaces = interfaces, .boundary = pack::TileBoundary::Last});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_interfaces(const pack::ReadInterfaces interfaces)
{
    TT_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        0 /*DstAccessMode*/,
        0 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        static_cast<std::uint32_t>(interfaces),
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        1 /*Last*/);
}

extern "C" __attribute__((noinline, used)) void run_runtime_address_modifier(const std::uint8_t address_modifier)
{
    pack::Packer0.transfer({.address_modifier = address_modifier, .dest_access = pack::DestAccess::Strided});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_address_modifier(const std::uint8_t address_modifier)
{
    TT_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        1 /*DstAccessMode*/,
        address_modifier,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        0 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        0 /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        0 /*Last*/);
}

// The fast-untilize row encoder selects concatenation and tile close at run time.
extern "C" __attribute__((noinline, used)) std::uint32_t encode_runtime_row(const bool concat, const bool last)
{
    return pack::Packer0.operation().transfer({
        .address_modifier = 1,
        .dest_access      = pack::DestAccess::Strided,
        .concatenation    = concat ? pack::Concatenation::Append : pack::Concatenation::NewRow,
        .boundary         = last ? pack::TileBoundary::Last : pack::TileBoundary::NotLast,
    });
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_runtime_row(const bool concat, const bool last)
{
    return TT_OP_PACR(
        0 /*CfgContext*/,
        0 /*RowPadZero*/,
        1 /*DstAccessMode*/,
        1 /*AddrMode*/,
        0 /*AddrCntContext*/,
        0 /*ZeroWrite*/,
        0 /*ReadIntfSel*/,
        0 /*OvrdThreadId*/,
        concat /*Concat*/,
        0 /*CtxtCtrl*/,
        0 /*Flush*/,
        last /*Last*/);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime operands must trap before the instruction is issued; valid
// boundaries must not trap.

extern "C" void check_runtime_address_modifier(const std::uint8_t address_modifier)
{
    pack::Packer0.transfer({.address_modifier = address_modifier});
}

// ASSERT-LABEL: <check_runtime_address_modifier>:
// ASSERT: ebreak

extern "C" void reject_invalid_address_modifier()
{
    pack::Packer0.transfer({.address_modifier = 4});
}

// ASSERT-LABEL: <reject_invalid_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_interfaces(const pack::ReadInterfaces interfaces)
{
    pack::Packer0.transfer({.interfaces = interfaces});
}

// ASSERT-LABEL: <check_runtime_interfaces>:
// ASSERT: ebreak

// 0b0110 is a 4-bit mask the Blackhole ISA does not support.
extern "C" void reject_unsupported_interfaces()
{
    pack::Packer0.transfer({.interfaces = static_cast<pack::ReadInterfaces>(0b0110)});
}

// ASSERT-LABEL: <reject_unsupported_interfaces>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_boundaries()
{
    pack::Packer0.transfer({.address_modifier = 3, .interfaces = pack::ReadInterfaces::Row1And3});
    pack::Packer0.transfer({.interfaces = pack::ReadInterfaces::All, .boundary = pack::TileBoundary::Last});
}

// ASSERT-LABEL: <accept_runtime_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret

// A branch-free remaining-rows mask, (1 << 4) - 1 for four rows, is the supported 0b1111.
extern "C" void accept_runtime_four_row_mask()
{
    pack::Packer0.transfer({.interfaces = static_cast<pack::ReadInterfaces>((1u << 4) - 1u)});
}

// ASSERT-LABEL: <accept_runtime_four_row_mask>:
// ASSERT-NOT: ebreak
// ASSERT: ret
