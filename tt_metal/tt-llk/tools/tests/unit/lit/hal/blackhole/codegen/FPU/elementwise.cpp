// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/fpu.h"

namespace fpu = hal::fpu;

// Elementwise operands print as clear_dvalid, dest_accum_en, broadcast, addr_mode, dst.

extern "C" __attribute__((noinline, used)) void elementwise_operations()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Subtract, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Multiply, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <elementwise_operations>:
// CHECK-NEXT: ttelwadd 0,0,0,0,0
// CHECK-NEXT: ttelwsub 0,0,0,0,0
// CHECK-NEXT: ttelwmul 0,0,0,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void elementwise_broadcasts()
{
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Column, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Row, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .broadcast = fpu::SrcBBroadcast::Scalar, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <elementwise_broadcasts>:
// CHECK-NEXT: ttelwadd 0,0,1,0,0
// CHECK-NEXT: ttelwadd 0,0,2,0,0
// CHECK-NEXT: ttelwadd 0,0,3,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void elementwise_options()
{
    fpu::run<fpu::Elementwise {
        .operation = fpu::ElementwiseOperation::Subtract, .dest_write = fpu::DestWriteMode::Accumulate, .release = fpu::SourceRelease::SrcA}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Add, .address_modifier = 7, .release = fpu::SourceRelease::SrcB}>();
    fpu::run<fpu::Elementwise {.operation = fpu::ElementwiseOperation::Multiply, .dest_row_offset = 1023, .release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::Elementwise {
        .operation        = fpu::ElementwiseOperation::Multiply,
        .broadcast        = fpu::SrcBBroadcast::Row,
        .dest_write       = fpu::DestWriteMode::Accumulate,
        .address_modifier = 3,
        .dest_row_offset  = 8,
        .release          = fpu::SourceRelease::Both}>();
}

// CHECK-LABEL: <elementwise_options>:
// CHECK-NEXT: ttelwsub 1,1,0,0,0
// CHECK-NEXT: ttelwadd 2,0,0,7,0
// CHECK-NEXT: ttelwmul 3,0,0,0,1023
// CHECK-NEXT: ttelwmul 3,1,2,3,8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_elementwise()
{
    return fpu::get_operation<fpu::Elementwise {
        .operation = fpu::ElementwiseOperation::Subtract, .broadcast = fpu::SrcBBroadcast::Scalar, .release = fpu::SourceRelease::SrcB}>();
}

// CHECK-LABEL: <encode_elementwise>:
// CHECK-NEXT: lui a0,0x30980
// CHECK-NEXT: ret

// Runtime descriptors are validated (invalid fields trap) and the selected
// opcode word is pushed through the instruction buffer.
extern "C" __attribute__((noinline, used)) void elementwise_runtime(const fpu::Elementwise operation)
{
    fpu::run(operation);
}

// CHECK-LABEL: <elementwise_runtime>:
// CHECK-DAG: lui {{a[0-7]}},0x28000
// CHECK-DAG: lui {{a[0-7]}},0x30000
// CHECK-DAG: lui {{a[0-7]}},0x27000
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK-DAG: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-DAG: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t encode_elementwise_runtime(const fpu::Elementwise operation)
{
    return fpu::get_operation(operation);
}

// CHECK-LABEL: <encode_elementwise_runtime>:
// CHECK-DAG: lui {{a[0-7]}},0x28000
// CHECK-DAG: lui {{a[0-7]}},0x30000
// CHECK-DAG: lui {{a[0-7]}},0x27000
// CHECK-DAG: ebreak
