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

#include "hal/src.h"

namespace src_ops = hal::src_ops;

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void transpose_sources()
{
    hal::src<hal::SrcA>.transpose();
    hal::src<hal::SrcB>.transpose();
}

// CHECK-LABEL: <transpose_sources>:
// CHECK-NEXT: tttrnspsrca{{$}}
// CHECK-NEXT: tttrnspsrcb{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void shift_srca_columns()
{
    hal::src<hal::SrcA>.shift_columns<src_ops::ShiftDirection::TowardColumn0>();
    hal::src<hal::SrcA>.shift_columns<src_ops::ShiftDirection::AwayFromColumn0>();
}

// CHECK-LABEL: <shift_srca_columns>:
// CHECK-NEXT: ttshiftxa 0,3{{$}}
// CHECK-NEXT: ttshiftxa 0,2{{$}}
// CHECK-NEXT: ret

// SHIFTXB operands print as addr_mode, rot_shift, shift_row; rot_shift 1 injects zero.
extern "C" __attribute__((noinline, used)) void shift_srcb_rows()
{
    hal::src<hal::SrcB>.shift_row<5>();
    hal::src<hal::SrcB>.shift_row<63, src_ops::ShiftFill::Rotate, 7>();
}

// CHECK-LABEL: <shift_srcb_rows>:
// CHECK-NEXT: ttshiftxb 0,1,5{{$}}
// CHECK-NEXT: ttshiftxb 7,0,63{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void rarefy_srcb()
{
    hal::src<hal::SrcB>.rarefy();
}

// CHECK-LABEL: <rarefy_srcb>:
// CHECK-NEXT: ttrareb{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_srca_right_shift_masks()
{
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal<0xffff, 1>();
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal<0x12>();
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal_thread0<3, 1>();
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal_thread1<5>();
    hal::src<hal::SrcA>.set_right_shift_mask_vertical<0xfffff>();
}

// CHECK-LABEL: <set_srca_right_shift_masks>:
// CHECK-NEXT: ttsetashrmh 65535,1{{$}}
// CHECK-NEXT: ttsetashrmh 18,0{{$}}
// CHECK-NEXT: ttsetashrmh0 3,1{{$}}
// CHECK-NEXT: ttsetashrmh1 5,0{{$}}
// CHECK-NEXT: ttsetashrmv 1048575{{$}}
// CHECK-NEXT: ret

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t encode_srcb_row_shift()
{
    return hal::src<hal::SrcB>.shift_row_operation<9, src_ops::ShiftFill::Rotate, 2>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_srcb_row_shift()
{
    return TT_OP_SHIFTXB(2 /*addr_mode*/, 0 /*rot_shift*/, 9 /*shift_row*/);
}

extern "C" __attribute__((noinline, used)) void shift_runtime_srcb_row(const std::uint32_t row)
{
    hal::src<hal::SrcB>.shift_row(row);
}

extern "C" __attribute__((noinline, used)) void reference_shift_runtime_srcb_row(const std::uint32_t row)
{
    TT_SHIFTXB(0 /*addr_mode*/, 1 /*rot_shift*/, row);
}

extern "C" __attribute__((noinline, used)) void shift_runtime_srcb_row_options(const std::uint32_t row)
{
    hal::src<hal::SrcB>.shift_row<src_ops::ShiftFill::Rotate, 7>(row);
}

extern "C" __attribute__((noinline, used)) void reference_shift_runtime_srcb_row_options(const std::uint32_t row)
{
    TT_SHIFTXB(7 /*addr_mode*/, 0 /*rot_shift*/, row);
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime rows must trap before the instruction is issued; valid
// boundaries must not trap.

extern "C" void check_runtime_srcb_row(const std::uint32_t row)
{
    hal::src<hal::SrcB>.shift_row(row);
}

// ASSERT-LABEL: <check_runtime_srcb_row>:
// ASSERT-NEXT: li [[LIMIT:a[0-7]]],63{{$}}
// ASSERT-NEXT: bltu [[LIMIT]],a0,
// ASSERT: ebreak

extern "C" void reject_invalid_runtime_srcb_row()
{
    hal::src<hal::SrcB>.shift_row(64);
}

// ASSERT-LABEL: <reject_invalid_runtime_srcb_row>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_srcb_row_boundary()
{
    hal::src<hal::SrcB>.shift_row(63);
}

// ASSERT-LABEL: <accept_runtime_srcb_row_boundary>:
// ASSERT-NOT: ebreak
// ASSERT: ret
