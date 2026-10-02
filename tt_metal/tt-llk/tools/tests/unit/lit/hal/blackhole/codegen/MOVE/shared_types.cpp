// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -dr %t.assert.o | FileCheck %s --check-prefix=ASSERT --enable-var-scope

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

extern "C" __attribute__((noinline, used)) std::uint32_t tile_row_runtime(const std::uint32_t tile_index, const std::uint32_t face, const std::uint32_t row)
{
    return move::dst_layout::tile_row(tile_index, face, row);
}

// CHECK-LABEL: <tile_row_runtime>:
// CHECK-NEXT: sh2add a0,a0,a1
// CHECK-NEXT: slli a0,a0,0x4
// CHECK-NEXT: add a0,a0,a2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t relative_row_runtime(const std::int32_t offset)
{
    return move::relative_row(offset);
}

// CHECK-LABEL: <relative_row_runtime>:
// CHECK-NEXT: andi a0,a0,1023
// CHECK-NEXT: ret

// ASSERT-LABEL: <relative_row_runtime>:
// ASSERT-NEXT: addi [[R0:a[0-7]]],a0,512
// ASSERT-NEXT: li [[R1:a[0-7]]],1023
// ASSERT-NEXT: bltu [[R1]],[[R0]],{{[0-9a-f]+}}
// ASSERT-NEXT: R_RISCV_BRANCH
// ASSERT-NEXT: andi a0,a0,1023
// ASSERT-NEXT: ret
// ASSERT-EMPTY:
// ASSERT-NEXT: <.L{{[0-9]+}}>:
// ASSERT-NEXT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t constant_tile_row()
{
    return move::dst_layout::tile_row(3, 2, 5);
}

// CHECK-LABEL: <constant_tile_row>:
// CHECK-NEXT: li a0,229
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t constant_relative_row()
{
    return move::relative_row<-1>();
}

// CHECK-LABEL: <constant_relative_row>:
// CHECK-NEXT: li a0,1023
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t constant_faces()
{
    return move::rows::faces<4>().value;
}

// CHECK-LABEL: <constant_faces>:
// CHECK-NEXT: li a0,64
// CHECK-NEXT: ret
