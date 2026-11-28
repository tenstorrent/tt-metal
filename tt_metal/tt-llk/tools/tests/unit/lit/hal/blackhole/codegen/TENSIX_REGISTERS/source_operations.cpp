// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump

// Compare each HAL function in codegen.cpp with its adjacent reference_
// function. Disable identical-code folding so both bodies are emitted
// independently.
// ZEROSRC operands print as (zero_val, write_mode, bank_mask, src_mask); CLEARDVALID as (cleardvalid, reset);
// GATESRCRST as (reset_srcb_gate_control, reset_srca_gate_control).

//--- common.h
#pragma once

#include <cstdint>

#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

//--- compile_time.cpp
#include "common.h"

// Zeroing writes zeros into the Matrix Unit's current bank of each selected register.
extern "C" __attribute__((noinline, used)) void zero_out_sources()
{
    tr::SrcA.zero_out();
    tr::SrcB.zero_out();
    (tr::SrcA | tr::SrcB).zero_out();
}

// CHECK-LABEL: <zero_out_sources>:
// CHECK-NEXT: ttzerosrc 0,1,0,1{{$}}
// CHECK-NEXT: ttzerosrc 0,1,0,2{{$}}
// CHECK-NEXT: ttzerosrc 0,1,0,3{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void release_sources()
{
    tr::SrcA.release();
    tr::SrcB.release();
    (tr::SrcA | tr::SrcB).release();
    (tr::SrcA | tr::SrcB).release<tr::BankAdvance::KeepReadingSame>();
}

// CHECK-LABEL: <release_sources>:
// CHECK-NEXT: ttcleardvalid 1,0{{$}}
// CHECK-NEXT: ttcleardvalid 2,0{{$}}
// CHECK-NEXT: ttcleardvalid 3,0{{$}}
// CHECK-NEXT: ttcleardvalid 3,2{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void single_register_operations()
{
    tr::SrcB.transpose();
    (tr::SrcA | tr::SrcB).reset_gate_control();
}

// CHECK-LABEL: <single_register_operations>:
// CHECK-NEXT: tttrnspsrcb{{$}}
// CHECK-NEXT: ttgatesrcrst 1,1{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

// Every source operation issues production's immediate.
extern "C" __attribute__((noinline, used)) void issue_source_operations()
{
    tr::SrcA.zero_out();
    (tr::SrcA | tr::SrcB).reset_gate_control();
    tr::SrcB.transpose();
    (tr::SrcA | tr::SrcB).reset_gate_control();
    tr::SrcB.release();
}

extern "C" __attribute__((noinline, used)) void reference_issue_source_operations()
{
    TTI_ZEROSRC(0 /*zero_val*/, 1 /*write_mode*/, 0 /*bank_mask*/, 1 /*src_mask*/);
    TTI_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
    TTI_TRNSPSRCB;
    TTI_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
    TTI_CLEARDVALID(0b10 /*cleardvalid*/, 0 /*reset*/);
}

// Encoded operations are plain words for MOP and replay programs.
extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    words[0] = tr::SrcA.operation().zero_out();
    words[1] = tr::SrcB.operation().transpose();
    words[2] = (tr::SrcA | tr::SrcB).operation().release();
    words[3] = tr::SrcB.operation().release<tr::BankAdvance::KeepReadingSame>();
    words[4] = (tr::SrcA | tr::SrcB).operation().reset_gate_control();
}

extern "C" __attribute__((noinline, used)) void reference_store_encoded_operations(std::uint32_t* words)
{
    words[0] = TT_OP_ZEROSRC(0 /*zero_val*/, 1 /*write_mode*/, 0 /*bank_mask*/, 1 /*src_mask*/);
    words[1] = TT_OP_TRNSPSRCB;
    words[2] = TT_OP_CLEARDVALID(0b11 /*cleardvalid*/, 0 /*reset*/);
    words[3] = TT_OP_CLEARDVALID(0b10 /*cleardvalid*/, 0b10 /*reset: keep reading the same bank*/);
    words[4] = TT_OP_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
}
