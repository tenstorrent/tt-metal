// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -UENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s

// Compare each runtime HAL function with its adjacent reference_ function.
// Disable identical-code folding so both bodies are emitted independently.
// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using GprToMmio = move::Transfer<move::Gpr, move::Mmio>;
using MmioToGpr = move::Transfer<move::Mmio, move::Gpr>;

//--- codegen.cpp
#include "common.h"

// TT_OP_ macros join fields with +, which compiles to add where the HAL's |
// compiles to or, so runtime operands are ORed onto an encoding that holds
// zero in their fields. RegAddr is the word offset from 0xffb00000:
// 0xffb11000 is 17408 and 0xffb12000 is 18432.

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_runtime_address(const std::uint32_t byte_address)
{
    move::run(GprToMmio {.source = hal::gpr<15>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = byte_address}});
}

extern "C" __attribute__((noinline, used)) void reference_gpr_to_mmio_runtime_address(const std::uint32_t byte_address)
{
    TT_INSN(TT_OP_STOREREG(15 /*TdmaDataRegIndex*/, 0 /*RegAddr*/) | ((byte_address - 0xffb00000u) >> 2));
}

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_runtime_addressing(const move::MmioAddressing addressing)
{
    move::run(GprToMmio {
        .source      = hal::gpr<16>(),
        .destination = {.addressing = addressing, .byte_address = 0xffb11000u, .base = hal::gpr<17>(), .offset = {.gpr = hal::gpr<18>()}},
    });
}

extern "C" __attribute__((noinline, used)) void reference_gpr_to_mmio_runtime_addressing(const move::MmioAddressing addressing)
{
    TT_INSN(
        addressing == move::MmioAddressing::Immediate
            ? TT_OP_STOREREG(16 /*TdmaDataRegIndex*/, 17408 /*RegAddr*/)
            : TT_OP_STOREIND(
                  0 /*MemHierSel*/, 1 /*SizeSel*/, 0 /*RegSizeSel*/, 36 /*OffsetIndex*/, 0 /*AutoIncSpec*/, 16 /*DataRegIndex*/, 17 /*AddrRegIndex*/));
}

extern "C" __attribute__((noinline, used)) void mmio_to_gpr_runtime_wait(const std::uint32_t destination)
{
    move::run(MmioToGpr {
        .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb12000u},
        .destination = hal::gpr(destination),
        .completion  = move::Completion::Wait,
    });
}

extern "C" __attribute__((noinline, used)) void reference_mmio_to_gpr_runtime_wait(const std::uint32_t destination)
{
    TT_INSN(TT_OP_LOADREG(0 /*TdmaDataRegIndex*/, 18432 /*RegAddr*/) | (destination << 18));
    TTI_STALLWAIT(511 /*stall_res*/, 1 /*wait_res*/);
}

//--- compile_time.cpp
#include "common.h"

constexpr GprToMmio store_first {
    .source      = hal::gpr<15>(),
    .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u},
};

constexpr GprToMmio store_last_wait {
    .source      = hal::gpr<63>(),
    .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffbffffcu},
    .completion  = move::Completion::Wait,
};

constexpr GprToMmio store_indirect {
    .source = hal::gpr<16>(),
    .destination =
        {
            .addressing       = move::MmioAddressing::Indirect,
            .base             = hal::gpr<17>(),
            .offset           = {.gpr = hal::gpr<18>(), .half = move::GprHalf::High},
            .offset_increment = move::OffsetIncrement::Bytes4,
        },
};

constexpr GprToMmio store_indirect_wait {
    .source      = hal::gpr<0>(),
    .destination = {.addressing = move::MmioAddressing::Indirect, .base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>()}},
    .completion  = move::Completion::Wait,
};

constexpr MmioToGpr load_last {
    .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffbffffcu},
    .destination = hal::gpr<19>(),
};

constexpr MmioToGpr load_first_wait {
    .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u},
    .destination = hal::gpr<0>(),
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_immediate()
{
    move::run<store_first>();
    move::run<store_last_wait>();
}

// CHECK-LABEL: <gpr_to_mmio_immediate>:
// CHECK-NEXT: ttstorereg 15,17408{{$}}
// CHECK-NEXT: ttstorereg 63,262143{{$}}
// CHECK-NEXT: ttstallwait 511,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void gpr_to_mmio_indirect()
{
    move::run<store_indirect>();
    move::run<store_indirect_wait>();
}

// CHECK-LABEL: <gpr_to_mmio_indirect>:
// CHECK-NEXT: ttstoreind 0,1,0,37,2,16,17{{$}}
// CHECK-NEXT: ttstoreind 0,1,0,4,0,0,1{{$}}
// CHECK-NEXT: ttstallwait 511,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mmio_to_gpr()
{
    move::run<load_last>();
    move::run<load_first_wait>();
}

// CHECK-LABEL: <mmio_to_gpr>:
// CHECK-NEXT: ttloadreg 19,262143{{$}}
// CHECK-NEXT: ttloadreg 0,17408{{$}}
// CHECK-NEXT: ttstallwait 511,1{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mmio_issue_encoded_operations()
{
    TTI_INSN(store_first.get_operation());
    TTI_INSN(store_indirect.get_operation());
    TTI_INSN(load_last.get_operation());
}

// CHECK-LABEL: <mmio_issue_encoded_operations>:
// CHECK-NEXT: ttstorereg 15,17408{{$}}
// CHECK-NEXT: ttstoreind 0,1,0,37,2,16,17{{$}}
// CHECK-NEXT: ttloadreg 19,262143{{$}}
// CHECK-NEXT: ret
