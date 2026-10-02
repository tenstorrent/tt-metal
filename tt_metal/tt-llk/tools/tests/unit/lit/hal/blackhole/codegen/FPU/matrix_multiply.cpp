// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/fpu.h"

namespace fpu = hal::fpu;

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
// CHECK-NEXT: ttmvmul 0,0,0,0
// CHECK-NEXT: ttmvmul 0,1,0,0
// CHECK-NEXT: ttmvmul 0,0,4,0
// CHECK-NEXT: ttmvmul 3,0,2,8
// CHECK-NEXT: ttmvmul 2,0,0,1023
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_matrix_multiply()
{
    return fpu::get_operation<fpu::MatrixMultiply {.broadcast = fpu::SrcBRowBroadcast::Row, .release = fpu::SourceRelease::SrcA}>();
}

// CHECK-LABEL: <encode_matrix_multiply>:
// CHECK-NEXT: lui a0,0x26480
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void matrix_multiply_runtime(const fpu::MatrixMultiply operation)
{
    fpu::run(operation);
}

// CHECK-LABEL: <matrix_multiply_runtime>:
// CHECK-DAG: lui {{a[0-7]}},0x26000
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK-DAG: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-DAG: ebreak
