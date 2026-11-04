// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/src.h"

namespace src_ops = hal::src_ops;

//--- compile_time.cpp
#include "common.h"

// ZEROSRC operands print as zero_val, write_mode, bank_mask, src_mask.

extern "C" __attribute__((noinline, used)) void fill_defaults()
{
    hal::src<hal::SrcA>.fill();
    hal::src<hal::SrcB>.fill();
    hal::src<hal::BothSources>.fill();
}

// CHECK-LABEL: <fill_defaults>:
// CHECK-NEXT: ttzerosrc 0,1,0,1{{$}}
// CHECK-NEXT: ttzerosrc 0,1,0,2{{$}}
// CHECK-NEXT: ttzerosrc 0,1,0,3{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void fill_options()
{
    hal::src<hal::SrcA>.fill<src_ops::BankScope::All>();
    hal::src<hal::SrcA>.fill<src_ops::BankScope::Current, src_ops::BankOwner::Unpacker>();
    hal::src<hal::SrcB>.fill<src_ops::BankScope::Current, src_ops::BankOwner::Math, src_ops::Fill::NegativeInfinity>();
    hal::src<hal::BothSources>.fill<src_ops::BankScope::All, src_ops::BankOwner::Unpacker, src_ops::Fill::NegativeInfinity>();
}

// CHECK-LABEL: <fill_options>:
// CHECK-NEXT: ttzerosrc 0,1,1,1{{$}}
// CHECK-NEXT: ttzerosrc 0,0,0,1{{$}}
// CHECK-NEXT: ttzerosrc 1,1,0,2{{$}}
// CHECK-NEXT: ttzerosrc 1,0,1,3{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void release_sources()
{
    hal::src<hal::SrcA>.release();
    hal::src<hal::SrcB>.release();
    hal::src<hal::BothSources>.release();
    hal::src<hal::BothSources>.release<src_ops::BankAdvance::KeepReadingSame>();
}

// CHECK-LABEL: <release_sources>:
// CHECK-NEXT: ttcleardvalid 1,0{{$}}
// CHECK-NEXT: ttcleardvalid 2,0{{$}}
// CHECK-NEXT: ttcleardvalid 3,0{{$}}
// CHECK-NEXT: ttcleardvalid 3,2{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void publish_sources()
{
    hal::src<hal::SrcA>.publish();
    hal::src<hal::SrcB>.publish();
    hal::src<hal::BothSources>.publish();
}

// CHECK-LABEL: <publish_sources>:
// CHECK-NEXT: ttsetdvalid 1{{$}}
// CHECK-NEXT: ttsetdvalid 2{{$}}
// CHECK-NEXT: ttsetdvalid 3{{$}}
// CHECK-NEXT: ret

// GATESRCRST operands print as SrcB reset, SrcA reset.
extern "C" __attribute__((noinline, used)) void reset_source_gating()
{
    hal::src<hal::SrcA>.reset_pipeline_gating();
    hal::src<hal::SrcB>.reset_pipeline_gating();
    hal::src<hal::BothSources>.reset_pipeline_gating();
}

// CHECK-LABEL: <reset_source_gating>:
// CHECK-NEXT: ttgatesrcrst 0,1{{$}}
// CHECK-NEXT: ttgatesrcrst 1,0{{$}}
// CHECK-NEXT: ttgatesrcrst 1,1{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_fill()
{
    return hal::src<hal::SrcB>.fill_operation<src_ops::BankScope::All, src_ops::BankOwner::Unpacker, src_ops::Fill::NegativeInfinity>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_fill()
{
    return TT_OP_ZEROSRC(1 /*zero_val*/, 0 /*write_mode*/, 1 /*bank_mask*/, 2 /*src_mask*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_release()
{
    return hal::src<hal::SrcA>.release_operation<src_ops::BankAdvance::KeepReadingSame>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_release()
{
    return TT_OP_CLEARDVALID(1 /*cleardvalid*/, 2 /*reset*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_publish()
{
    return hal::src<hal::BothSources>.publish_operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_publish()
{
    return TT_OP_SETDVALID(3 /*setvalid*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_gating_reset()
{
    return hal::src<hal::SrcB>.reset_pipeline_gating_operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_gating_reset()
{
    return TT_OP_GATESRCRST(1 /*reset_srcb_gate_control*/, 0 /*reset_srca_gate_control*/);
}
