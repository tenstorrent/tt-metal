// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// codegen.cpp is compiled without LLK_ASSERT so runtime paths contain only the
// issued instruction; asserts.cpp checks the runtime operand asserts.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/fpu.h"

namespace fpu = hal::fpu;

//--- compile_time.cpp
#include "common.h"

// Elementwise operands print as clear_dvalid, dest_accum_en, broadcast, addr_mode, dst.

extern "C" __attribute__((noinline, used)) void elementwise_operations()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Subtract, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Multiply, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <elementwise_operations>:
// CHECK-NEXT: ttelwadd 0,0,0,0,0{{$}}
// CHECK-NEXT: ttelwsub 0,0,0,0,0{{$}}
// CHECK-NEXT: ttelwmul 0,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void elementwise_broadcasts()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Column, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Row, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Scalar, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <elementwise_broadcasts>:
// CHECK-NEXT: ttelwadd 0,0,1,0,0{{$}}
// CHECK-NEXT: ttelwadd 0,0,2,0,0{{$}}
// CHECK-NEXT: ttelwadd 0,0,3,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void elementwise_options()
{
    fpu::run<fpu::Elementwise {
        .operation = fpu::ElementwiseOperation::Subtract, .dest_write = fpu::DestWriteMode::Accumulate, .release = fpu::SourceRelease::SrcA}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .address_modifier = 7, .release = fpu::SourceRelease::SrcB}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Multiply, .dest_row_offset = 1023, .release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::Elementwise {
        .operation        = fpu::ElementwiseOperation::Multiply,
        .broadcast        = fpu::SrcBBroadcast::Row,
        .dest_write       = fpu::DestWriteMode::Accumulate,
        .address_modifier = 3,
        .dest_row_offset  = 8,
        .release          = fpu::SourceRelease::Both}>();
}

// CHECK-LABEL: <elementwise_options>:
// CHECK-NEXT: ttelwsub 1,1,0,0,0{{$}}
// CHECK-NEXT: ttelwadd 2,0,0,7,0{{$}}
// CHECK-NEXT: ttelwmul 3,0,0,0,1023{{$}}
// CHECK-NEXT: ttelwmul 3,1,2,3,8{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_elementwise()
{
    return fpu::get_operation<fpu::Elementwise {
        .operation = fpu::ElementwiseOperation::Subtract, .broadcast = fpu::SrcBBroadcast::Scalar, .release = fpu::SourceRelease::SrcB}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_elementwise()
{
    return TT_OP_ELWSUB(2 /*clear_dvalid*/, 0 /*dest_accum_en*/, 3 /*instr_mod19*/, 0 /*addr_mode*/, 0 /*dst*/);
}

inline __attribute__((always_inline)) std::uint32_t reference_elementwise_operation(const fpu::Elementwise operation)
{
    const std::uint32_t clear_dvalid  = static_cast<std::uint32_t>(operation.release);
    const std::uint32_t dest_accum_en = static_cast<std::uint32_t>(operation.dest_write);
    const std::uint32_t broadcast     = static_cast<std::uint32_t>(operation.broadcast);
    switch (operation.operation)
    {
        case fpu::ElementwiseOperation::Add:
            return TT_OP_ELWADD(clear_dvalid, dest_accum_en, broadcast, operation.address_modifier, operation.dest_row_offset);
        case fpu::ElementwiseOperation::Subtract:
            return TT_OP_ELWSUB(clear_dvalid, dest_accum_en, broadcast, operation.address_modifier, operation.dest_row_offset);
        default:
            return TT_OP_ELWMUL(clear_dvalid, dest_accum_en, broadcast, operation.address_modifier, operation.dest_row_offset);
    }
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_elementwise_runtime(const fpu::Elementwise operation)
{
    return fpu::get_operation(operation);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_elementwise_runtime(const fpu::Elementwise operation)
{
    return reference_elementwise_operation(operation);
}

// Three TT_ELW* pushes in the switch arms tail-merge differently from one push
// of the selected word, so the reference issues the word with TT_INSN.
extern "C" __attribute__((noinline, used)) void elementwise_runtime(const fpu::Elementwise operation)
{
    fpu::run(operation);
}

extern "C" __attribute__((noinline, used)) void reference_elementwise_runtime(const fpu::Elementwise operation)
{
    TT_INSN(reference_elementwise_operation(operation));
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime descriptors must trap before the instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_operation()
{
    fpu::run(fpu::Elementwise {.operation = static_cast<fpu::ElementwiseOperation>(3), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_operation>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_broadcast()
{
    fpu::run(
        fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = static_cast<fpu::SrcBBroadcast>(4), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_broadcast>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_write()
{
    fpu::run(
        fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .dest_write = static_cast<fpu::DestWriteMode>(2), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_write>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_address_modifier()
{
    fpu::run(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .address_modifier = 8, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_row_offset()
{
    fpu::run(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .dest_row_offset = 1024, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_row_offset>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unset_runtime_release()
{
    fpu::run(fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add});
}

// ASSERT-LABEL: <reject_unset_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_elementwise_boundaries()
{
    fpu::run(fpu::Elementwise {
        .operation        = fpu::ElementwiseOperation::Multiply,
        .broadcast        = fpu::SrcBBroadcast::Scalar,
        .dest_write       = fpu::DestWriteMode::Accumulate,
        .address_modifier = 7,
        .dest_row_offset  = 1023,
        .release          = fpu::SourceRelease::Both});
}

// ASSERT-LABEL: <accept_runtime_elementwise_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
