// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/fpu.h"

namespace fpu = hal::fpu;

// Pool operands print as clear_dvalid, instr_mod19, addr_mode, index_en, dst;
// instr_mod19 is always encoded as 1.

extern "C" __attribute__((noinline, used)) void pool_functions()
{
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <pool_functions>:
// CHECK-NEXT: ttgapool 0,1,0,0,0
// CHECK-NEXT: ttgmpool 0,1,0,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void pool_options()
{
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .address_modifier = 7, .release = fpu::SourceRelease::None}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum, .dest_row_offset = 16, .release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Maximum, .indices = fpu::IndexTracking::Enabled, .release = fpu::SourceRelease::SrcA}>();
    fpu::run<fpu::Pool {.function = fpu::PoolFunction::Sum, .dest_row_offset = 1023, .release = fpu::SourceRelease::SrcB}>();
}

// CHECK-LABEL: <pool_options>:
// CHECK-NEXT: ttgapool 0,1,7,0,0
// CHECK-NEXT: ttgmpool 3,1,0,0,16
// CHECK-NEXT: ttgmpool 1,1,0,1,0
// CHECK-NEXT: ttgapool 2,1,0,0,1023
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_pool()
{
    return fpu::get_operation<fpu::Pool {
        .function = fpu::PoolFunction::Maximum, .indices = fpu::IndexTracking::Enabled, .address_modifier = 1, .release = fpu::SourceRelease::None}>();
}

// CHECK-LABEL: <encode_pool>:
// CHECK-NEXT: lui a0,0x3308c
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_pool_runtime(const fpu::Pool operation)
{
    return fpu::get_operation(operation);
}

// CHECK-LABEL: <encode_pool_runtime>:
// CHECK-DAG: lui {{a[0-7]}},0x34080
// CHECK-DAG: lui {{a[0-7]}},0x33080
// CHECK-DAG: ebreak

extern "C" __attribute__((noinline, used)) void pool_runtime(const fpu::Pool operation)
{
    fpu::run(operation);
}

// CHECK-LABEL: <pool_runtime>:
// CHECK-DAG: lui {{a[0-7]}},0x34080
// CHECK-DAG: lui {{a[0-7]}},0x33080
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK-DAG: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-DAG: ebreak
