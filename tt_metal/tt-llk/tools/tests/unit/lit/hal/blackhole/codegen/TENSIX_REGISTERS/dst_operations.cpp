// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

// Compare each HAL function in codegen.cpp with its adjacent reference_
// function. Disable identical-code folding so both bodies are emitted
// independently. codegen.cpp is compiled without LLK_ASSERT so runtime paths
// contain only the issued instruction; asserts.cpp checks the runtime operand asserts.
// ZEROACC operands print as (clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where).

//--- common.h
#pragma once

#include <cstdint>

#include "hal/tensix_registers.h"

namespace tr = hal::tensix_registers;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void mark_invalid_scopes()
{
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::SingleRow, .index = 100}>();
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::Face, .index = 3}>();
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::Half, .index = 1}>();
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::All}>();
}

// CHECK-LABEL: <mark_invalid_scopes>:
// CHECK-NEXT: ttzeroacc 0,0,0,0,100{{$}}
// CHECK-NEXT: ttzeroacc 1,0,0,0,3{{$}}
// CHECK-NEXT: ttzeroacc 2,0,0,0,1{{$}}
// CHECK-NEXT: ttzeroacc 3,0,0,0,0{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mark_invalid_options()
{
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::SingleRow, .index = 1023, .address_mode = 7}>();
    tr::Dst.mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::All, .width = tr::DestWidth::Bits32, .address_mode = 1}>();
}

// CHECK-LABEL: <mark_invalid_options>:
// CHECK-NEXT: ttzeroacc 0,0,0,7,1023{{$}}
// CHECK-NEXT: ttzeroacc 3,1,0,1,0{{$}}
// CHECK-NEXT: ret

// Marking a face valid again is the 16-row mode with Revert set.
extern "C" __attribute__((noinline, used)) void mark_valid_faces()
{
    tr::Dst.mark_valid<tr::DstMarkValid {.face = 5, .address_mode = 3}>();
    tr::Dst.mark_valid<tr::DstMarkValid {.face = 1023, .width = tr::DestWidth::Bits16, .address_mode = 3}>();
}

// CHECK-LABEL: <mark_valid_faces>:
// CHECK-NEXT: ttzeroacc 1,1,1,3,5{{$}}
// CHECK-NEXT: ttzeroacc 1,0,1,3,1023{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    words[0] = tr::Dst.operation().mark_invalid<tr::DstMarkInvalid {.scope = tr::ZeroScope::Face, .index = 2, .width = tr::DestWidth::Bits32}>();
    words[1] = tr::Dst.operation().mark_valid<tr::DstMarkValid {.face = 4, .address_mode = 3}>();
}

extern "C" __attribute__((noinline, used)) void reference_store_encoded_operations(std::uint32_t* words)
{
    words[0] = TT_OP_ZEROACC(1 /*clear_mode*/, 1 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, 0 /*addr_mode*/, 2 /*where*/);
    words[1] = TT_OP_ZEROACC(1 /*clear_mode*/, 1 /*use_32_bit_mode*/, 1 /*clear_zero_flags*/, 3 /*addr_mode*/, 4 /*where*/);
}

// Runtime options are encoded in a register and pushed through the instruction buffer.
extern "C" __attribute__((noinline, used)) void mark_invalid_runtime_half(const std::uint32_t half, const bool fp32)
{
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::Half, .index = half, .width = fp32 ? tr::DestWidth::Bits32 : tr::DestWidth::Bits16, .address_mode = 1});
}

extern "C" __attribute__((noinline, used)) void reference_mark_invalid_runtime_half(const std::uint32_t half, const bool fp32)
{
    TT_ZEROACC(2 /*clear_mode*/, fp32 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, 1 /*addr_mode*/, half /*where*/);
}

extern "C" __attribute__((noinline, used)) void mark_valid_runtime_face(const std::uint32_t face, const bool fp32)
{
    tr::Dst.mark_valid({.face = face, .width = fp32 ? tr::DestWidth::Bits32 : tr::DestWidth::Bits16, .address_mode = 3});
}

extern "C" __attribute__((noinline, used)) void reference_mark_valid_runtime_face(const std::uint32_t face, const bool fp32)
{
    TT_ZEROACC(1 /*clear_mode*/, fp32 /*use_32_bit_mode*/, 1 /*clear_zero_flags*/, 3 /*addr_mode*/, face /*where*/);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime options must trap before the instruction is issued; valid
// boundaries must not trap.

extern "C" void reject_invalid_scope()
{
    tr::Dst.mark_invalid({.scope = static_cast<tr::ZeroScope>(4)});
}

// ASSERT-LABEL: <reject_invalid_scope>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

// The row, face or half selector is the 10-bit Imm10.
extern "C" void reject_single_row_index()
{
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::SingleRow, .index = 1024});
}

// ASSERT-LABEL: <reject_single_row_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_half_index()
{
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::Half, .index = 2});
}

// ASSERT-LABEL: <reject_half_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_all_index()
{
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::All, .index = 1});
}

// ASSERT-LABEL: <reject_all_index>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_address_mode()
{
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::All, .address_mode = 8});
}

// ASSERT-LABEL: <reject_address_mode>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_mark_valid_face()
{
    tr::Dst.mark_valid({.face = 1024});
}

// ASSERT-LABEL: <reject_mark_valid_face>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void check_runtime_face(const std::uint32_t face)
{
    tr::Dst.mark_valid({.face = face, .address_mode = 3});
}

// ASSERT-LABEL: <check_runtime_face>:
// ASSERT: ebreak

extern "C" void accept_runtime_boundaries()
{
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::SingleRow, .index = 1023, .address_mode = 7});
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::Half, .index = 1});
    tr::Dst.mark_invalid({.scope = tr::ZeroScope::All});
    tr::Dst.mark_valid({.face = 1023, .width = tr::DestWidth::Bits16, .address_mode = 7});
}

// ASSERT-LABEL: <accept_runtime_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
