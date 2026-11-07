// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s

// Compare each runtime HAL function with its adjacent reference_ function.
// Disable identical-code folding so both bodies are emitted independently.
// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using L1ToGpr   = move::Transfer<move::L1, move::Gpr>;
using GprToL1   = move::Transfer<move::Gpr, move::L1>;
using GprToSrcA = move::Transfer<move::Gpr, move::SrcA>;
using GprToSrcB = move::Transfer<move::Gpr, move::SrcB>;

//--- codegen.cpp
#include "common.h"

// TT_OP_ macros join fields with +, which compiles to add where the HAL's |
// compiles to or, so runtime operands are ORed onto an encoding that holds
// zero in their fields. OffsetIndex selects a 16-bit GPR half: 2 * gpr + (half == High).

extern "C" __attribute__((noinline, used)) void l1_to_gpr_runtime(const std::uint32_t base, const std::uint32_t offset, const std::uint32_t destination)
{
    move::run(L1ToGpr {
        .source =
            {
                .base             = hal::gpr(base),
                .offset           = {.gpr = hal::gpr(offset), .half = move::GprHalf::High},
                .offset_increment = move::OffsetIncrement::Bytes4,
            },
        .destination = hal::gpr(destination),
        .size        = move::ScalarSize::Bytes4,
    });
}

extern "C" __attribute__((noinline, used)) void reference_l1_to_gpr_runtime(
    const std::uint32_t base, const std::uint32_t offset, const std::uint32_t destination)
{
    TT_INSN(
        TT_OP_LOADIND(1 /*SizeSel*/, 0 /*OffsetIndex*/, 2 /*AutoIncSpec*/, 0 /*DataRegIndex*/, 0 /*AddrRegIndex*/) | base | (destination << 6) |
        ((2 * offset + 1) << 14));
}

extern "C" __attribute__((noinline, used)) void gpr_to_l1_runtime_completion(const std::uint32_t source, const move::Completion completion)
{
    move::run(GprToL1 {
        .source      = hal::gpr(source),
        .destination = {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<6>()}},
        .size        = move::ScalarSize::Bytes2,
        .completion  = completion,
    });
}

extern "C" __attribute__((noinline, used)) void reference_gpr_to_l1_runtime_completion(const std::uint32_t source, const move::Completion completion)
{
    TT_INSN(
        TT_OP_STOREIND(1 /*MemHierSel*/, 1 /*SizeSel*/, 0 /*RegSizeSel*/, 12 /*OffsetIndex*/, 0 /*AutoIncSpec*/, 0 /*DataRegIndex*/, 5 /*AddrRegIndex*/) |
        (source << 6));
    if (completion == move::Completion::Wait)
    {
        TTI_STALLWAIT(511 /*stall_res*/, 1 /*wait_res*/);
    }
}

extern "C" __attribute__((noinline, used)) void gpr_to_srcb_runtime(const std::uint32_t source)
{
    move::run(GprToSrcB {.source = hal::gpr(source), .destination = {.base = hal::gpr<13>(), .offset = {.gpr = hal::gpr<14>()}}});
}

extern "C" __attribute__((noinline, used)) void reference_gpr_to_srcb_runtime(const std::uint32_t source)
{
    TT_INSN(
        TT_OP_STOREIND(0 /*MemHierSel*/, 0 /*SizeSel*/, 1 /*RegSizeSel*/, 28 /*OffsetIndex*/, 0 /*AutoIncSpec*/, 0 /*DataRegIndex*/, 13 /*AddrRegIndex*/) |
        (source << 6));
}

extern "C" __attribute__((noinline, used)) std::uint32_t l1_to_gpr_encode_runtime(const std::uint32_t destination)
{
    return L1ToGpr {.source = {.base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>()}}, .destination = hal::gpr(destination), .size = move::ScalarSize::Bytes1}
        .get_operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_l1_to_gpr_encode_runtime(const std::uint32_t destination)
{
    return TT_OP_LOADIND(3 /*SizeSel*/, 4 /*OffsetIndex*/, 0 /*AutoIncSpec*/, 0 /*DataRegIndex*/, 1 /*AddrRegIndex*/) | (destination << 6);
}

//--- compile_time.cpp
#include "common.h"

template <move::ScalarSize Size, move::OffsetIncrement Increment, move::GprHalf Half>
constexpr L1ToGpr load {
    .source      = {.base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>(), .half = Half}, .offset_increment = Increment},
    .destination = hal::gpr<4>(),
    .size        = Size,
};

extern "C" __attribute__((noinline, used)) void l1_to_gpr_sizes()
{
    move::run<load<move::ScalarSize::Bytes16, move::OffsetIncrement::None, move::GprHalf::Low>>();
    move::run<load<move::ScalarSize::Bytes4, move::OffsetIncrement::Bytes4, move::GprHalf::High>>();
    move::run<load<move::ScalarSize::Bytes2, move::OffsetIncrement::Bytes2, move::GprHalf::Low>>();
    move::run<load<move::ScalarSize::Bytes1, move::OffsetIncrement::Bytes16, move::GprHalf::High>>();
}

// CHECK-LABEL: <l1_to_gpr_sizes>:
// CHECK-NEXT: ttloadind 0,4,0,4,1{{$}}
// CHECK-NEXT: ttloadind 1,5,2,4,1{{$}}
// CHECK-NEXT: ttloadind 2,4,1,4,1{{$}}
// CHECK-NEXT: ttloadind 3,5,3,4,1{{$}}
// CHECK-NEXT: ret

constexpr L1ToGpr load_wait {
    .source      = {.base = hal::gpr<63>(), .offset = {.gpr = hal::gpr<63>(), .half = move::GprHalf::High}},
    .destination = hal::gpr<60>(),
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void l1_to_gpr_wait()
{
    move::run<load_wait>();
}

// CHECK-LABEL: <l1_to_gpr_wait>:
// CHECK-NEXT: ttloadind 0,127,0,60,63{{$}}
// CHECK-NEXT: ttstallwait 511,1{{$}}
// CHECK-NEXT: ret

template <move::ScalarSize Size, move::OffsetIncrement Increment>
constexpr GprToL1 store {
    .source      = hal::gpr<8>(),
    .destination = {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<6>(), .half = move::GprHalf::Low}, .offset_increment = Increment},
    .size        = Size,
};

extern "C" __attribute__((noinline, used)) void gpr_to_l1_sizes()
{
    move::run<store<move::ScalarSize::Bytes16, move::OffsetIncrement::Bytes16>>();
    move::run<store<move::ScalarSize::Bytes4, move::OffsetIncrement::None>>();
    move::run<store<move::ScalarSize::Bytes2, move::OffsetIncrement::Bytes2>>();
    move::run<store<move::ScalarSize::Bytes1, move::OffsetIncrement::Bytes4>>();
}

// CHECK-LABEL: <gpr_to_l1_sizes>:
// CHECK-NEXT: ttstoreind 1,0,0,12,3,8,5{{$}}
// CHECK-NEXT: ttstoreind 1,0,1,12,0,8,5{{$}}
// CHECK-NEXT: ttstoreind 1,1,0,12,1,8,5{{$}}
// CHECK-NEXT: ttstoreind 1,1,1,12,2,8,5{{$}}
// CHECK-NEXT: ret

constexpr GprToL1 store_wait {
    .source      = hal::gpr<4>(),
    .destination = {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<6>(), .half = move::GprHalf::Low}, .offset_increment = move::OffsetIncrement::Bytes16},
    .size        = move::ScalarSize::Bytes2,
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void gpr_to_l1_wait()
{
    move::run<store_wait>();
}

// CHECK-LABEL: <gpr_to_l1_wait>:
// CHECK-NEXT: ttstoreind 1,1,0,12,3,4,5{{$}}
// CHECK-NEXT: ttstallwait 511,1{{$}}
// CHECK-NEXT: ret

constexpr GprToSrcA store_srca {
    .source      = hal::gpr<8>(),
    .destination = {.base = hal::gpr<9>(), .offset = {.gpr = hal::gpr<10>(), .half = move::GprHalf::High}, .offset_increment = move::OffsetIncrement::Bytes2},
};

constexpr GprToSrcB store_srcb {
    .source      = hal::gpr<12>(),
    .destination = {.base = hal::gpr<13>(), .offset = {.gpr = hal::gpr<14>(), .half = move::GprHalf::Low}},
};

extern "C" __attribute__((noinline, used)) void gpr_to_src()
{
    move::run<store_srca>();
    move::run<store_srcb>();
}

// CHECK-LABEL: <gpr_to_src>:
// CHECK-NEXT: ttstoreind 0,0,0,21,1,8,9{{$}}
// CHECK-NEXT: ttstoreind 0,0,1,28,0,12,13{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void scalar_issue_encoded_operations()
{
    TTI_INSN((load<move::ScalarSize::Bytes4, move::OffsetIncrement::Bytes4, move::GprHalf::High>.get_operation()));
    TTI_INSN((store<move::ScalarSize::Bytes1, move::OffsetIncrement::Bytes4>.get_operation()));
    TTI_INSN(store_srca.get_operation());
    TTI_INSN(store_srcb.get_operation());
}

// CHECK-LABEL: <scalar_issue_encoded_operations>:
// CHECK-NEXT: ttloadind 1,5,2,4,1{{$}}
// CHECK-NEXT: ttstoreind 1,1,1,12,2,8,5{{$}}
// CHECK-NEXT: ttstoreind 0,0,0,21,1,8,9{{$}}
// CHECK-NEXT: ttstoreind 0,0,1,28,0,12,13{{$}}
// CHECK-NEXT: ret
