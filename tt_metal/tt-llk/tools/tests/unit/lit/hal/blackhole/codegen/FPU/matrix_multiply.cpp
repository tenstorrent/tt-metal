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

// MVMUL operands print as clear_dvalid, broadcast, addr_mode, dst.

extern "C" __attribute__((noinline, used)) void matrix_multiplies()
{
    fpu::run<fpu::MatrixMultiply {.release = fpu::SourceRelease::None}>();
    fpu::run<fpu::MatrixMultiply {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::MatrixMultiply {.address_modifier = 4, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::MatrixMultiply {.address_modifier = 2, .dest_row_offset = 8, .release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::MatrixMultiply {.dest_row_offset = 1023, .release = fpu::SourceRelease::SrcB}>();
}

// CHECK-LABEL: <matrix_multiplies>:
// CHECK-NEXT: ttmvmul 0,0,0,0{{$}}
// CHECK-NEXT: ttmvmul 0,1,0,0{{$}}
// CHECK-NEXT: ttmvmul 0,0,4,0{{$}}
// CHECK-NEXT: ttmvmul 3,0,2,8{{$}}
// CHECK-NEXT: ttmvmul 2,0,0,1023{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_matrix_multiply()
{
    return fpu::get_operation<fpu::MatrixMultiply {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::SrcA}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_matrix_multiply()
{
    return TT_OP_MVMUL(1 /*clear_dvalid*/, 1 /*instr_mod19*/, 0 /*addr_mode*/, 0 /*dst*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_matrix_multiply_runtime(const fpu::MatrixMultiply operation)
{
    return fpu::get_operation(operation);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_matrix_multiply_runtime(const fpu::MatrixMultiply operation)
{
    return TT_OP_MVMUL(
        static_cast<std::uint32_t>(operation.release), static_cast<std::uint32_t>(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
}

extern "C" __attribute__((noinline, used)) void matrix_multiply_runtime(const fpu::MatrixMultiply operation)
{
    fpu::run(operation);
}

extern "C" __attribute__((noinline, used)) void reference_matrix_multiply_runtime(const fpu::MatrixMultiply operation)
{
    TT_MVMUL(
        static_cast<std::uint32_t>(operation.release), static_cast<std::uint32_t>(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime descriptors must trap before the instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_broadcast()
{
    fpu::run(fpu::MatrixMultiply {.broadcast = static_cast<fpu::SrcBRowBroadcast>(2), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_broadcast>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_address_modifier()
{
    fpu::run(fpu::MatrixMultiply {.address_modifier = 8, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_row_offset()
{
    fpu::run(fpu::MatrixMultiply {.dest_row_offset = 1024, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_row_offset>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unset_runtime_release()
{
    fpu::run(fpu::MatrixMultiply {});
}

// ASSERT-LABEL: <reject_unset_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_matrix_multiply_boundaries()
{
    fpu::run(
        fpu::MatrixMultiply {.broadcast = fpu::SrcBRowBroadcast::Row, .address_modifier = 7, .dest_row_offset = 1023, .release = fpu::SourceRelease::Both});
}

// ASSERT-LABEL: <accept_runtime_matrix_multiply_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
