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

// DOTPV operands print as clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst;
// the interface always encodes the two reserved fields as zero.

extern "C" __attribute__((noinline, used)) void dot_products()
{
    fpu::run<fpu::DotProduct {.release = fpu::SourceRelease::None}>();
    fpu::run<fpu::DotProduct {.release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::DotProduct {.address_modifier = 7, .dest_row_offset = 1023, .release = fpu::SourceRelease::SrcA}>();
}

// CHECK-LABEL: <dot_products>:
// CHECK-NEXT: ttdotpv 0,0,0,0,0{{$}}
// CHECK-NEXT: ttdotpv 3,0,0,0,0{{$}}
// CHECK-NEXT: ttdotpv 1,0,0,7,1023{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_dot_product()
{
    return fpu::get_operation<fpu::DotProduct {.address_modifier = 1, .release = fpu::SourceRelease::SrcB}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_dot_product()
{
    return TT_OP_DOTPV(2 /*clear_dvalid*/, 0 /*dest_accum_en*/, 0 /*instr_mod19*/, 1 /*addr_mode*/, 0 /*dst*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_dot_product_runtime(const fpu::DotProduct operation)
{
    return fpu::get_operation(operation);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_dot_product_runtime(const fpu::DotProduct operation)
{
    return TT_OP_DOTPV(
        static_cast<std::uint32_t>(operation.release), 0 /*dest_accum_en*/, 0 /*instr_mod19*/, operation.address_modifier, operation.dest_row_offset);
}

extern "C" __attribute__((noinline, used)) void dot_product_runtime(const fpu::DotProduct operation)
{
    fpu::run(operation);
}

extern "C" __attribute__((noinline, used)) void reference_dot_product_runtime(const fpu::DotProduct operation)
{
    TT_DOTPV(static_cast<std::uint32_t>(operation.release), 0 /*dest_accum_en*/, 0 /*instr_mod19*/, operation.address_modifier, operation.dest_row_offset);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime descriptors must trap before the instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_address_modifier()
{
    fpu::run(fpu::DotProduct {.address_modifier = 8, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_address_modifier>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_dest_row_offset()
{
    fpu::run(fpu::DotProduct {.dest_row_offset = 1024, .release = fpu::SourceRelease::None});
}

// ASSERT-LABEL: <reject_invalid_runtime_dest_row_offset>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_unset_runtime_release()
{
    fpu::run(fpu::DotProduct {});
}

// ASSERT-LABEL: <reject_unset_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_dot_product_boundaries()
{
    fpu::run(fpu::DotProduct {.address_modifier = 7, .dest_row_offset = 1023, .release = fpu::SourceRelease::Both});
}

// ASSERT-LABEL: <accept_runtime_dot_product_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
