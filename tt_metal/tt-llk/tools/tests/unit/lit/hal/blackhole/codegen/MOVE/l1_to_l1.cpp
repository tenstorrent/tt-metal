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

// THCON mover 0 words: 88 = source, 89 = destination, 90 = size | direction 3 << 30.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using L1ToL1 = move::Transfer<move::L1, move::L1>;

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void l1_to_l1_runtime(
    const std::uint32_t source, const std::uint32_t destination, const std::uint32_t size, const move::Completion completion)
{
    move::run(L1ToL1 {.source = {.block = source}, .destination = {.block = destination}, .size = {.value = size}, .completion = completion});
}

extern "C" __attribute__((noinline, used)) void reference_l1_to_l1_runtime(
    const std::uint32_t source, const std::uint32_t destination, const std::uint32_t size, const move::Completion completion)
{
    TT_RMWCIB0(0xff /*Mask*/, source & 0xffu, 88 /*CfgRegAddr*/);
    TT_RMWCIB1(0xff /*Mask*/, (source >> 8) & 0xffu, 88 /*CfgRegAddr*/);
    TT_RMWCIB2(0xff /*Mask*/, (source >> 16) & 0xffu, 88 /*CfgRegAddr*/);
    TT_RMWCIB3(0xff /*Mask*/, source >> 24, 88 /*CfgRegAddr*/);
    TT_RMWCIB0(0xff /*Mask*/, destination & 0xffu, 89 /*CfgRegAddr*/);
    TT_RMWCIB1(0xff /*Mask*/, (destination >> 8) & 0xffu, 89 /*CfgRegAddr*/);
    TT_RMWCIB2(0xff /*Mask*/, (destination >> 16) & 0xffu, 89 /*CfgRegAddr*/);
    TT_RMWCIB3(0xff /*Mask*/, destination >> 24, 89 /*CfgRegAddr*/);
    const std::uint32_t size_direction = size | (3u << 30);
    TT_RMWCIB0(0xff /*Mask*/, size_direction & 0xffu, 90 /*CfgRegAddr*/);
    TT_RMWCIB1(0xff /*Mask*/, (size_direction >> 8) & 0xffu, 90 /*CfgRegAddr*/);
    TT_RMWCIB2(0xff /*Mask*/, (size_direction >> 16) & 0xffu, 90 /*CfgRegAddr*/);
    TT_RMWCIB3(0xff /*Mask*/, size_direction >> 24, 90 /*CfgRegAddr*/);
    TTI_XMOV(0 /*Mov_block_selection*/, 0 /*Last*/);
    if (completion == move::Completion::Wait)
    {
        TTI_STALLWAIT(511 /*stall_res*/, 512 /*wait_res*/);
    }
}

//--- compile_time.cpp
#include "common.h"

constexpr L1ToL1 copy {
    .source      = {.block = 0x10000},
    .destination = {.block = 0},
    .size        = {.value = 0x8000},
};

constexpr L1ToL1 copy_wait {
    .source      = {.block = 0x100},
    .destination = {.block = 0x200},
    .size        = {.value = 16},
    .completion  = move::Completion::Wait,
};

extern "C" __attribute__((noinline, used)) void l1_to_l1_issue_only()
{
    move::run<copy>();
}

// CHECK-LABEL: <l1_to_l1_issue_only>:
// CHECK-NEXT: ttrmwcib0 255,0,88{{$}}
// CHECK-NEXT: ttrmwcib1 255,0,88{{$}}
// CHECK-NEXT: ttrmwcib2 255,1,88{{$}}
// CHECK-NEXT: ttrmwcib3 255,0,88{{$}}
// CHECK-NEXT: ttrmwcib0 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib1 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib2 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib3 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib0 255,0,90{{$}}
// CHECK-NEXT: ttrmwcib1 255,128,90{{$}}
// CHECK-NEXT: ttrmwcib2 255,0,90{{$}}
// CHECK-NEXT: ttrmwcib3 255,192,90{{$}}
// CHECK-NEXT: ttxmov 0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void l1_to_l1_wait()
{
    move::run<copy_wait>();
}

// CHECK-LABEL: <l1_to_l1_wait>:
// CHECK-NEXT: ttrmwcib0 255,0,88{{$}}
// CHECK-NEXT: ttrmwcib1 255,1,88{{$}}
// CHECK-NEXT: ttrmwcib2 255,0,88{{$}}
// CHECK-NEXT: ttrmwcib3 255,0,88{{$}}
// CHECK-NEXT: ttrmwcib0 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib1 255,2,89{{$}}
// CHECK-NEXT: ttrmwcib2 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib3 255,0,89{{$}}
// CHECK-NEXT: ttrmwcib0 255,16,90{{$}}
// CHECK-NEXT: ttrmwcib1 255,0,90{{$}}
// CHECK-NEXT: ttrmwcib2 255,0,90{{$}}
// CHECK-NEXT: ttrmwcib3 255,192,90{{$}}
// CHECK-NEXT: ttxmov 0,0{{$}}
// CHECK-NEXT: ttstallwait 511,512{{$}}
// CHECK-NEXT: ret
