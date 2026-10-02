// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/dst.h"

namespace dst = hal::dst;

// ZEROACC operands print as clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where.

extern "C" __attribute__((noinline, used)) void zero_scopes()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 100}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::Face, .index = 3}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .index = 1}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::All}>();
}

// CHECK-LABEL: <zero_scopes>:
// CHECK-NEXT: ttzeroacc 0,0,0,0,100
// CHECK-NEXT: ttzeroacc 1,0,0,0,3
// CHECK-NEXT: ttzeroacc 2,0,0,0,1
// CHECK-NEXT: ttzeroacc 3,0,0,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void zero_options()
{
    dst::run<dst::Zero {.scope = dst::ZeroScope::Face, .index = 255, .flags = dst::ZeroFlagAction::ClearFlags}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .width = dst::DestWidth::Bits32}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::SingleRow, .index = 0x3fff, .address_mode = 7}>();
    dst::run<dst::Zero {.scope = dst::ZeroScope::All, .flags = dst::ZeroFlagAction::ClearFlags, .width = dst::DestWidth::Bits32, .address_mode = 5}>();
}

// CHECK-LABEL: <zero_options>:
// CHECK-NEXT: ttzeroacc 1,0,1,0,255
// CHECK-NEXT: ttzeroacc 2,1,0,0,0
// CHECK-NEXT: ttzeroacc 0,0,0,7,16383
// CHECK-NEXT: ttzeroacc 3,1,1,5,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_zero()
{
    constexpr std::uint32_t operation = dst::Zero {.scope = dst::ZeroScope::Face, .index = 2, .width = dst::DestWidth::Bits32}.get_operation();
    return operation;
}

// CHECK-LABEL: <encode_zero>:
// CHECK-NEXT: lui a0,0x100c0
// CHECK-NEXT: addi a0,a0,2
// CHECK-NEXT: ret

// The runtime descriptor is validated (invalid fields trap) and its ZEROACC word
// is built in a register and pushed through the instruction buffer.
extern "C" __attribute__((noinline, used)) void zero_runtime(const dst::Zero operation)
{
    dst::run(operation);
}

// CHECK-LABEL: <zero_runtime>:
// CHECK: lui {{a[0-7]}},0x10000
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret
// CHECK: ebreak
