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

#include "hal/dst.h"

namespace dst = hal::dst;

//--- compile_time.cpp
#include "common.h"

// ZEROACC operands print as clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where.

extern "C" __attribute__((noinline, used)) void zero_scopes()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 100}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::Face, .index = 3}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .index = 1}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::All}>();
}

// CHECK-LABEL: <zero_scopes>:
// CHECK-NEXT: ttzeroacc 0,0,0,0,100{{$}}
// CHECK-NEXT: ttzeroacc 1,0,0,0,3{{$}}
// CHECK-NEXT: ttzeroacc 2,0,0,0,1{{$}}
// CHECK-NEXT: ttzeroacc 3,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void zero_options()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::Face, .index = 255, .flags = dst::ZeroFlagAction::ClearFlags}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .width = dst::DestWidth::Bits32}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 0x3fff, .address_mode = 7}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::All, .flags = dst::ZeroFlagAction::ClearFlags, .width = dst::DestWidth::Bits32, .address_mode = 5}>();
}

// CHECK-LABEL: <zero_options>:
// CHECK-NEXT: ttzeroacc 1,0,1,0,255{{$}}
// CHECK-NEXT: ttzeroacc 2,1,0,0,0{{$}}
// CHECK-NEXT: ttzeroacc 0,0,0,7,16383{{$}}
// CHECK-NEXT: ttzeroacc 3,1,1,5,0{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_zero()
{
    constexpr std::uint32_t operation = dst::Zero {.scope = dst::ZeroScope::Face, .index = 2, .width = dst::DestWidth::Bits32}.get_operation();
    return operation;
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_zero()
{
    return TT_OP_ZEROACC(1 /*clear_mode*/, 1 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, 0 /*addr_mode*/, 2 /*where*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_zero_runtime(const dst::Zero operation)
{
    return operation.get_operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_zero_runtime(const dst::Zero operation)
{
    return TT_OP_ZEROACC(
        static_cast<std::uint32_t>(operation.scope),
        operation.width == dst::DestWidth::Bits32 ? 1u : 0u,
        operation.flags == dst::ZeroFlagAction::ClearFlags ? 1u : 0u,
        operation.address_mode,
        operation.index);
}

extern "C" __attribute__((noinline, used)) void zero_runtime(const dst::Zero operation)
{
    dst::run(operation);
}

extern "C" __attribute__((noinline, used)) void reference_zero_runtime(const dst::Zero operation)
{
    TT_ZEROACC(
        static_cast<std::uint32_t>(operation.scope),
        operation.width == dst::DestWidth::Bits32 ? 1u : 0u,
        operation.flags == dst::ZeroFlagAction::ClearFlags ? 1u : 0u,
        operation.address_mode,
        operation.index);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime descriptors must trap before the instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_scope()
{
    dst::run(dst::Zero {.scope = static_cast<dst::ZeroScope>(4)});
}

// ASSERT-LABEL: <reject_invalid_runtime_scope>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_single_row_index()
{
    dst::run(dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 0x4000});
}

// ASSERT-LABEL: <reject_invalid_runtime_single_row_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_face_index()
{
    dst::run(dst::Zero {.scope = dst::ZeroScope::Face, .index = 256});
}

// ASSERT-LABEL: <reject_invalid_runtime_face_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_half_index()
{
    dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = 2});
}

// ASSERT-LABEL: <reject_invalid_runtime_half_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_all_index()
{
    dst::run(dst::Zero {.scope = dst::ZeroScope::All, .index = 1});
}

// ASSERT-LABEL: <reject_invalid_runtime_all_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_address_mode()
{
    dst::run(dst::Zero {.scope = dst::ZeroScope::SingleRow, .address_mode = 8});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_mode>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_zero_boundaries()
{
    dst::run(dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 0x3fff, .address_mode = 7});
    dst::run(dst::Zero {.scope = dst::ZeroScope::Face, .index = 255});
    dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = 1});
    dst::run(dst::Zero {.scope = dst::ZeroScope::All});
}

// ASSERT-LABEL: <accept_runtime_zero_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
