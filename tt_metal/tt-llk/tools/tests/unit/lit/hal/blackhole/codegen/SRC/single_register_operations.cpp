// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/src.h"

namespace src_ops = hal::src_ops;

// objdump has no TRNSPSRCA mnemonic.
extern "C" __attribute__((noinline, used)) void transpose_sources()
{
    hal::src<hal::SrcA>.transpose();
    hal::src<hal::SrcB>.transpose();
}

// CHECK-LABEL: <transpose_sources>:
// CHECK-NEXT: .insn 4, 0x50000000
// CHECK-NEXT: tttrnspsrcb
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void shift_srca_columns()
{
    hal::src<hal::SrcA>.shift_columns<src_ops::ShiftDirection::TowardColumn0>();
    hal::src<hal::SrcA>.shift_columns<src_ops::ShiftDirection::AwayFromColumn0>();
}

// CHECK-LABEL: <shift_srca_columns>:
// CHECK-NEXT: ttshiftxa 0,3
// CHECK-NEXT: ttshiftxa 0,2
// CHECK-NEXT: ret

// SHIFTXB operands print as addr_mode, rot_shift, shift_row; rot_shift 1 injects zero.
extern "C" __attribute__((noinline, used)) void shift_srcb_rows()
{
    hal::src<hal::SrcB>.shift_row<5>();
    hal::src<hal::SrcB>.shift_row<63, src_ops::ShiftFill::Rotate, 7>();
}

// CHECK-LABEL: <shift_srcb_rows>:
// CHECK-NEXT: ttshiftxb 0,1,5
// CHECK-NEXT: ttshiftxb 7,0,63
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_srcb_row_shift()
{
    return hal::src<hal::SrcB>.shift_row_operation<9, src_ops::ShiftFill::Rotate, 2>();
}

// CHECK-LABEL: <encode_srcb_row_shift>:
// CHECK-NEXT: lui a0,0x18008
// CHECK-NEXT: addi a0,a0,9
// CHECK-NEXT: ret

// The runtime row joins the SHIFTXB word (0x18000400: zero fill, address mode 0)
// and is pushed through the instruction buffer; rows past 63 trap.
extern "C" __attribute__((noinline, used)) void shift_runtime_srcb_row(std::uint32_t row)
{
    hal::src<hal::SrcB>.shift_row(row);
}

// CHECK-LABEL: <shift_runtime_srcb_row>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,{{.*}}
// CHECK: lui [[OP:a[0-7]]],0x18000
// CHECK-NEXT: addi [[OP]],[[OP]],1024
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add [[WORD:a[0-7]]],a0,[[OP]]
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void rarefy_srcb()
{
    hal::src<hal::SrcB>.rarefy();
}

// CHECK-LABEL: <rarefy_srcb>:
// CHECK-NEXT: ttrareb
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
// CHECK-NEXT: ttsetashrmh 65535,1
// CHECK-NEXT: ttsetashrmh 18,0
// CHECK-NEXT: ttsetashrmh0 3,1
// CHECK-NEXT: ttsetashrmh1 5,0
// CHECK-NEXT: ttsetashrmv 1048575
// CHECK-NEXT: ret
