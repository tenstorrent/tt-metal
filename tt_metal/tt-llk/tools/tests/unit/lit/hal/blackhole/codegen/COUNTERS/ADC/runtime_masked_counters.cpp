// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace adc      = hal::counters::adc;

// Preserve the original masked-input case and its instruction expectations.
// Masked arithmetic can reassociate differently in the HAL and raw TT_* wrappers;
// runtime_counters.cpp compares unmasked operands byte-for-byte against TT_*.
extern "C" __attribute__((noinline, used)) void runtime_advance_and_reset_xy(std::uint32_t x, std::uint32_t y)
{
    const auto xy_reload =
        adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = {.x = x & 0x7u, .y = y & 0x7u}}.for_clients<adc::Client::Unpacker0>();
    counters::set(xy_reload);
}

// ADDRCRXY, client 1, write mask 0b0011.
// CHECK-LABEL: <runtime_advance_and_reset_xy>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x53200
// CHECK-DAG: addi [[OP]],[[OP]],3
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret
