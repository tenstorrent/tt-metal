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

// Eltwise operands print as clear_dvalid, dest_accum_en, broadcast, addr_mode, dst.

extern "C" __attribute__((noinline, used)) void eltwise_operations()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Subtract, .release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Multiply, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <eltwise_operations>:
// CHECK-NEXT: ttelwadd 0,0,0,0,0{{$}}
// CHECK-NEXT: ttelwsub 0,0,0,0,0{{$}}
// CHECK-NEXT: ttelwmul 0,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void eltwise_broadcasts()
{
    fpu::MatrixUnit
        .eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Column, .release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit
        .eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Row, .release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit
        .eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Scalar, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <eltwise_broadcasts>:
// CHECK-NEXT: ttelwadd 0,0,1,0,0{{$}}
// CHECK-NEXT: ttelwadd 0,0,2,0,0{{$}}
// CHECK-NEXT: ttelwadd 0,0,3,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void eltwise_options()
{
    fpu::MatrixUnit.eltwise<fpu::Eltwise {
        .operation = fpu::EltwiseOperation::Subtract, .dest_write = fpu::DestWriteMode::Accumulate, .release = fpu::SourceRelease::SrcA}>();
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .address_modifier = 7, .release = fpu::SourceRelease::SrcB}>();
    fpu::MatrixUnit.eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Multiply, .dest_row_offset = 1023, .release = fpu::SourceRelease::Both}>();
    fpu::MatrixUnit.eltwise<fpu::Eltwise {
        .operation        = fpu::EltwiseOperation::Multiply,
        .broadcast        = fpu::SrcBBroadcast::Row,
        .dest_write       = fpu::DestWriteMode::Accumulate,
        .address_modifier = 3,
        .dest_row_offset  = 8,
        .release          = fpu::SourceRelease::Both}>();
}

// CHECK-LABEL: <eltwise_options>:
// CHECK-NEXT: ttelwsub 1,1,0,0,0{{$}}
// CHECK-NEXT: ttelwadd 2,0,0,7,0{{$}}
// CHECK-NEXT: ttelwmul 3,0,0,0,1023{{$}}
// CHECK-NEXT: ttelwmul 3,1,2,3,8{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_eltwise()
{
    return fpu::MatrixUnit.operation()
        .eltwise<fpu::Eltwise {.operation = fpu::EltwiseOperation::Subtract, .broadcast = fpu::SrcBBroadcast::Scalar, .release = fpu::SourceRelease::SrcB}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_eltwise()
{
    return TT_OP_ELWSUB(2 /*clear_dvalid*/, 0 /*dest_accum_en*/, 3 /*instr_mod19*/, 0 /*addr_mode*/, 0 /*dst*/);
}

inline __attribute__((always_inline)) std::uint32_t reference_eltwise_operation(const fpu::Eltwise operation)
{
    const std::uint32_t clear_dvalid  = static_cast<std::uint32_t>(operation.release);
    const std::uint32_t dest_accum_en = static_cast<std::uint32_t>(operation.dest_write);
    const std::uint32_t broadcast     = static_cast<std::uint32_t>(operation.broadcast);
    switch (operation.operation)
    {
        case fpu::EltwiseOperation::Add:
            return TT_OP_ELWADD(clear_dvalid, dest_accum_en, broadcast, operation.address_modifier, operation.dest_row_offset);
        case fpu::EltwiseOperation::Subtract:
            return TT_OP_ELWSUB(clear_dvalid, dest_accum_en, broadcast, operation.address_modifier, operation.dest_row_offset);
        default:
            return TT_OP_ELWMUL(clear_dvalid, dest_accum_en, broadcast, operation.address_modifier, operation.dest_row_offset);
    }
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_eltwise_runtime(const fpu::Eltwise operation)
{
    return fpu::MatrixUnit.operation().eltwise(operation);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_eltwise_runtime(const fpu::Eltwise operation)
{
    return reference_eltwise_operation(operation);
}

// Three TT_ELW* pushes in the switch arms tail-merge differently from one push
// of the selected word, so the reference issues the word with TT_INSN.
extern "C" __attribute__((noinline, used)) void eltwise_runtime(const fpu::Eltwise operation)
{
    fpu::MatrixUnit.eltwise(operation);
}

extern "C" __attribute__((noinline, used)) void reference_eltwise_runtime(const fpu::Eltwise operation)
{
    TT_INSN(reference_eltwise_operation(operation));
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime descriptors must trap before the instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_operation()
{
    fpu::MatrixUnit.eltwise(fpu::Eltwise {.operation = static_cast<fpu::EltwiseOperation>(3), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_operation>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_broadcast()
{
    fpu::MatrixUnit.eltwise(
        fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .broadcast = static_cast<fpu::SrcBBroadcast>(4), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_broadcast>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_write()
{
    fpu::MatrixUnit.eltwise(
        fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .dest_write = static_cast<fpu::DestWriteMode>(2), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_write>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_address_modifier()
{
    fpu::MatrixUnit.eltwise(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .address_modifier = 8, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_row_offset()
{
    fpu::MatrixUnit.eltwise(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add, .dest_row_offset = 1024, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_row_offset>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unset_runtime_release()
{
    fpu::MatrixUnit.eltwise(fpu::Eltwise {.operation = fpu::EltwiseOperation::Add});
}

// ASSERT-LABEL: <reject_unset_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_eltwise_boundaries()
{
    fpu::MatrixUnit.eltwise(fpu::Eltwise {
        .operation        = fpu::EltwiseOperation::Multiply,
        .broadcast        = fpu::SrcBBroadcast::Scalar,
        .dest_write       = fpu::DestWriteMode::Accumulate,
        .address_modifier = 7,
        .dest_row_offset  = 1023,
        .release          = fpu::SourceRelease::Both});
}

// ASSERT-LABEL: <accept_runtime_eltwise_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
