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

// Pool operands print as clear_dvalid, instr_mod19, addr_mode, index_en, dst;
// instr_mod19 is always encoded as 1.

extern "C" __attribute__((noinline, used)) void pool_functions()
{
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <pool_functions>:
// CHECK-NEXT: ttgapool 0,1,0,0,0{{$}}
// CHECK-NEXT: ttgmpool 0,1,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void pool_options()
{
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .address_modifier = 7, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum, .dest_row_offset = 16, .release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::SrcA}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .dest_row_offset = 1023, .release = fpu::SourceRelease::SrcB}>();
}

// CHECK-LABEL: <pool_options>:
// CHECK-NEXT: ttgapool 0,1,7,0,0{{$}}
// CHECK-NEXT: ttgmpool 3,1,0,0,16{{$}}
// CHECK-NEXT: ttgmpool 1,1,0,1,0{{$}}
// CHECK-NEXT: ttgapool 2,1,0,0,1023{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_pool()
{
    return fpu::get_operation<fpu::Pool {
        .function = fpu::PoolFunction::Maximum, .indices = fpu::IndexTracking::Enabled, .address_modifier = 1, .release = fpu::SourceRelease::None}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_pool()
{
    return TT_OP_GMPOOL(0 /*clear_dvalid*/, 1 /*instr_mod19*/, 1 /*pool_addr_mode*/, 1 /*max_pool_index_en*/, 0 /*dst*/);
}

inline __attribute__((always_inline)) std::uint32_t reference_pool_operation(const fpu::Pool operation)
{
    const std::uint32_t clear_dvalid      = static_cast<std::uint32_t>(operation.release);
    const std::uint32_t max_pool_index_en = static_cast<std::uint32_t>(operation.indices);
    return operation.function == fpu::PoolFunction::Sum
               ? TT_OP_GAPOOL(clear_dvalid, 1 /*instr_mod19*/, operation.address_modifier, max_pool_index_en, operation.dest_row_offset)
               : TT_OP_GMPOOL(clear_dvalid, 1 /*instr_mod19*/, operation.address_modifier, max_pool_index_en, operation.dest_row_offset);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_pool_runtime(const fpu::Pool operation)
{
    return fpu::get_operation(operation);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_pool_runtime(const fpu::Pool operation)
{
    return reference_pool_operation(operation);
}

// Separate TT_GAPOOL/TT_GMPOOL pushes in each branch are not merged into one
// push of the selected word, so the reference issues the word with TT_INSN.
extern "C" __attribute__((noinline, used)) void pool_runtime(const fpu::Pool operation)
{
    fpu::run(operation);
}

extern "C" __attribute__((noinline, used)) void reference_pool_runtime(const fpu::Pool operation)
{
    TT_INSN(reference_pool_operation(operation));
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime descriptors must trap before the instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_function()
{
    fpu::run(fpu::Pool {.function = static_cast<fpu::PoolFunction>(2), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_function>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_indices()
{
    fpu::run(fpu::Pool {.function = fpu::PoolFunction::Maximum, .indices = static_cast<fpu::IndexTracking>(2), .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_indices>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_address_modifier()
{
    fpu::run(fpu::Pool {.function = fpu::PoolFunction::Sum, .address_modifier = 8, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_row_offset()
{
    fpu::run(fpu::Pool {.function = fpu::PoolFunction::Sum, .dest_row_offset = 1024, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_row_offset>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unset_runtime_release()
{
    fpu::run(fpu::Pool {.function = fpu::PoolFunction::Sum});
}

// ASSERT-LABEL: <reject_unset_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_runtime_sum_index_tracking()
{
    fpu::run(fpu::Pool {.function = fpu::PoolFunction::Sum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_runtime_sum_index_tracking>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_pool_boundaries()
{
    fpu::run(fpu::Pool {.function = fpu::PoolFunction::Sum, .address_modifier = 7, .dest_row_offset = 1023, .release = fpu::SourceRelease::Both});
    fpu::run(fpu::Pool {
        .function         = fpu::PoolFunction::Maximum,
        .indices          = fpu::IndexTracking::Enabled,
        .address_modifier = 7,
        .dest_row_offset  = 1023,
        .release          = fpu::SourceRelease::Both});
}

// ASSERT-LABEL: <accept_runtime_pool_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
