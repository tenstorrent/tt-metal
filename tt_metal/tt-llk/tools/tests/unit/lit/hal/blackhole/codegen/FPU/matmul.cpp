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

extern "C" __attribute__((noinline, used)) void matmuls()
{
    fpu::MatrixUnit.matmul<fpu::Matmul {.release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit.matmul<fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit.matmul<fpu::Matmul {.address_modifier = 4, .release = fpu::SourceRelease::None}>();
    fpu::MatrixUnit.matmul<fpu::Matmul {.address_modifier = 2, .dest_row_offset = 8, .release = fpu::SourceRelease::Both}>();
    fpu::MatrixUnit.matmul<fpu::Matmul {.dest_row_offset = 1023, .release = fpu::SourceRelease::SrcB}>();
}

// CHECK-LABEL: <matmuls>:
// CHECK-NEXT: ttmvmul 0,0,0,0{{$}}
// CHECK-NEXT: ttmvmul 0,1,0,0{{$}}
// CHECK-NEXT: ttmvmul 0,0,4,0{{$}}
// CHECK-NEXT: ttmvmul 3,0,2,8{{$}}
// CHECK-NEXT: ttmvmul 2,0,0,1023{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_matmul()
{
    return fpu::MatrixUnit.operation().matmul<fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::SrcA}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_matmul()
{
    return TT_OP_MVMUL(1 /*clear_dvalid*/, 1 /*instr_mod19*/, 0 /*addr_mode*/, 0 /*dst*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_matmul_runtime(const fpu::Matmul operation)
{
    return fpu::MatrixUnit.operation().matmul(operation);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_matmul_runtime(const fpu::Matmul operation)
{
    return TT_OP_MVMUL(
        static_cast<std::uint32_t>(operation.release), static_cast<std::uint32_t>(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
}

extern "C" __attribute__((noinline, used)) void matmul_runtime(const fpu::Matmul operation)
{
    fpu::MatrixUnit.matmul(operation);
}

extern "C" __attribute__((noinline, used)) void reference_matmul_runtime(const fpu::Matmul operation)
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
    fpu::MatrixUnit.matmul(fpu::Matmul {.broadcast = static_cast<fpu::SrcBRowBroadcast>(2), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_broadcast>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_address_modifier()
{
    fpu::MatrixUnit.matmul(fpu::Matmul {.address_modifier = 8, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_row_offset()
{
    fpu::MatrixUnit.matmul(fpu::Matmul {.dest_row_offset = 1024, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_row_offset>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

// Row broadcast with Dst row bit 1 or 2 set is UnsupportedFunctionality.
extern "C" void reject_misaligned_runtime_row_broadcast()
{
    fpu::MatrixUnit.matmul(fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .dest_row_offset = 2, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_misaligned_runtime_row_broadcast>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unset_runtime_release()
{
    fpu::MatrixUnit.matmul(fpu::Matmul {});
}

// ASSERT-LABEL: <reject_unset_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_matmul_boundaries()
{
    fpu::MatrixUnit.matmul(
        fpu::Matmul {.broadcast = fpu::SrcBRowBroadcast::Row, .address_modifier = 7, .dest_row_offset = 1016, .release = fpu::SourceRelease::Both});
}

// ASSERT-LABEL: <accept_runtime_matmul_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
