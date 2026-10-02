// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include <cstdint>

#include "hal/src.h"

namespace src_ops = hal::src_ops;

// ZEROSRC operands print as zero_val, write_mode, bank_mask, src_mask.

extern "C" __attribute__((noinline, used)) void fill_defaults()
{
    hal::src<hal::SrcA>.fill();
    hal::src<hal::SrcB>.fill();
    hal::src<hal::BothSources>.fill();
}

// CHECK-LABEL: <fill_defaults>:
// CHECK-NEXT: ttzerosrc 0,1,0,1
// CHECK-NEXT: ttzerosrc 0,1,0,2
// CHECK-NEXT: ttzerosrc 0,1,0,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void fill_options()
{
    hal::src<hal::SrcA>.fill<src_ops::BankScope::All>();
    hal::src<hal::SrcA>.fill<src_ops::BankScope::Current, src_ops::BankOwner::Unpacker>();
    hal::src<hal::SrcB>.fill<src_ops::BankScope::Current, src_ops::BankOwner::Math, src_ops::Fill::NegativeInfinity>();
    hal::src<hal::BothSources>.fill<src_ops::BankScope::All, src_ops::BankOwner::Unpacker, src_ops::Fill::NegativeInfinity>();
}

// CHECK-LABEL: <fill_options>:
// CHECK-NEXT: ttzerosrc 0,1,1,1
// CHECK-NEXT: ttzerosrc 0,0,0,1
// CHECK-NEXT: ttzerosrc 1,1,0,2
// CHECK-NEXT: ttzerosrc 1,0,1,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_fill()
{
    return hal::src<hal::SrcB>.fill_operation<src_ops::BankScope::All, src_ops::BankOwner::Unpacker, src_ops::Fill::NegativeInfinity>();
}

// CHECK-LABEL: <encode_fill>:
// CHECK-NEXT: lui a0,0x11000
// CHECK-NEXT: addi a0,a0,22
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void release_sources()
{
    hal::src<hal::SrcA>.release();
    hal::src<hal::SrcB>.release();
    hal::src<hal::BothSources>.release();
    hal::src<hal::BothSources>.release<src_ops::BankAdvance::KeepReadingSame>();
}

// CHECK-LABEL: <release_sources>:
// CHECK-NEXT: ttcleardvalid 1,0
// CHECK-NEXT: ttcleardvalid 2,0
// CHECK-NEXT: ttcleardvalid 3,0
// CHECK-NEXT: ttcleardvalid 3,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_release()
{
    return hal::src<hal::SrcA>.release_operation<src_ops::BankAdvance::KeepReadingSame>();
}

// CHECK-LABEL: <encode_release>:
// CHECK-NEXT: lui a0,0x36400
// CHECK-NEXT: addi a0,a0,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void publish_sources()
{
    hal::src<hal::SrcA>.publish();
    hal::src<hal::SrcB>.publish();
    hal::src<hal::BothSources>.publish();
}

// CHECK-LABEL: <publish_sources>:
// CHECK-NEXT: ttsetdvalid 1
// CHECK-NEXT: ttsetdvalid 2
// CHECK-NEXT: ttsetdvalid 3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_publish()
{
    return hal::src<hal::BothSources>.publish_operation();
}

// CHECK-LABEL: <encode_publish>:
// CHECK-NEXT: lui a0,0x57000
// CHECK-NEXT: addi a0,a0,3
// CHECK-NEXT: ret

// GATESRCRST operands print as SrcB reset, SrcA reset.
extern "C" __attribute__((noinline, used)) void reset_source_gating()
{
    hal::src<hal::SrcA>.reset_pipeline_gating();
    hal::src<hal::SrcB>.reset_pipeline_gating();
    hal::src<hal::BothSources>.reset_pipeline_gating();
}

// CHECK-LABEL: <reset_source_gating>:
// CHECK-NEXT: ttgatesrcrst 0,1
// CHECK-NEXT: ttgatesrcrst 1,0
// CHECK-NEXT: ttgatesrcrst 1,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_gating_reset()
{
    return hal::src<hal::SrcB>.reset_pipeline_gating_operation();
}

// CHECK-LABEL: <encode_gating_reset>:
// CHECK-NEXT: lui a0,0x35000
// CHECK-NEXT: addi a0,a0,2
// CHECK-NEXT: ret
