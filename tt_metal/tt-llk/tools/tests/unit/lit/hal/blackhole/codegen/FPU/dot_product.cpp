// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/fpu.h"

namespace fpu = hal::fpu;

// DOTPV operands print as clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst;
// the interface always encodes the two reserved fields as zero.

extern "C" __attribute__((noinline, used)) void dot_products()
{
    fpu::run<fpu::DotProduct {.release = fpu::SourceRelease::None}>();
    fpu::run<fpu::DotProduct {.release = fpu::SourceRelease::Both}>();
    fpu::run<fpu::DotProduct {.address_modifier = 7, .dest_row_offset = 1023, .release = fpu::SourceRelease::SrcA}>();
}

// CHECK-LABEL: <dot_products>:
// CHECK-NEXT: ttdotpv 0,0,0,0,0
// CHECK-NEXT: ttdotpv 3,0,0,0,0
// CHECK-NEXT: ttdotpv 1,0,0,7,1023
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_dot_product()
{
    return fpu::get_operation<fpu::DotProduct {.address_modifier = 1, .release = fpu::SourceRelease::SrcB}>();
}

// CHECK-LABEL: <encode_dot_product>:
// CHECK-NEXT: lui a0,0x29804
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dot_product_runtime(const fpu::DotProduct operation)
{
    fpu::run(operation);
}

// CHECK-LABEL: <dot_product_runtime>:
// CHECK-DAG: lui {{a[0-7]}},0x29000
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK-DAG: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-DAG: ebreak
