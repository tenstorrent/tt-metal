// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/math_counters.h"

using hal::rwc::Counters;

extern "C" __attribute__((noinline, used)) void set_shared_value()
{
    hal::math_counters.set<Counters::SrcA | Counters::SrcB, 4>().apply();
}

// CHECK-LABEL: <set_shared_value>:
// CHECK-NEXT: ttsetrwc 0,0,0,4,4,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_distinct_values()
{
    hal::math_counters.set<Counters::SrcA, 1>().set<Counters::SrcB, 2>().set<Counters::Dest, 3>().apply();
}

// CHECK-LABEL: <set_distinct_values>:
// CHECK-NEXT: ttsetrwc 0,0,3,2,1,7
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_shared_value()
{
    hal::math_counters.increment<Counters::SrcA | Counters::SrcB, 4>().apply();
}

// CHECK-LABEL: <increment_shared_value>:
// CHECK-NEXT: ttincrwc 0,0,4,4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_and_reload_from_carry()
{
    hal::math_counters.set<Counters::SrcA | Counters::SrcB, 4>().advance_carry_and_reload<Counters::SrcA | Counters::SrcB>().apply();
}

// CHECK-LABEL: <set_and_reload_from_carry>:
// CHECK-NEXT: ttsetrwc 0,3,0,4,4,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_dest_and_reload_from_carry()
{
    hal::math_counters.increment<Counters::Dest, 2>().advance_carry_and_reload<Counters::Dest>().apply();
}

// CHECK-LABEL: <increment_dest_and_reload_from_carry>:
// CHECK-NEXT: ttincrwc 4,2,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void release_sources_and_clear_fidelity()
{
    hal::math_counters.release<Counters::SrcA | Counters::SrcB>().clear_fidelity().apply();
}

// CHECK-LABEL: <release_sources_and_clear_fidelity>:
// CHECK-NEXT: ttsetrwc 3,0,0,0,0,8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void advance_dest_and_save_to_carry()
{
    hal::math_counters.advance_dest_and_save_to_carry<7>().apply();
}

// CHECK-LABEL: <advance_dest_and_save_to_carry>:
// CHECK-NEXT: ttsetrwc 0,8,7,0,0,4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_bias()
{
    hal::math_counters.set<Counters::Bias, 0xabc>().apply();
}

// CHECK-LABEL: <set_bias>:
// CHECK-NEXT: ttsetibrwc 0,2748,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_bias_and_reload_from_carry()
{
    hal::math_counters.increment<Counters::Bias, 0x123>().advance_carry_and_reload<Counters::Bias>().apply();
}

// CHECK-LABEL: <increment_bias_and_reload_from_carry>:
// CHECK-NEXT: ttsetibrwc 1,291,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_sources_and_bias()
{
    hal::math_counters.set<Counters::SrcA | Counters::SrcB, 4>().set<Counters::Bias, 0x123>().apply();
}

// CHECK-LABEL: <set_sources_and_bias>:
// CHECK-NEXT: ttsetrwc 0,0,0,4,4,3
// CHECK-NEXT: ttsetibrwc 0,291,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_dest_and_bias()
{
    hal::math_counters.increment<Counters::Dest, 8>().increment<Counters::Bias, 1>().apply();
}

// CHECK-LABEL: <increment_dest_and_bias>:
// CHECK-NEXT: ttincrwc 0,8,0,0
// CHECK-NEXT: ttsetibrwc 0,1,1
// CHECK-NEXT: ret

// get_operation() returns the encoded word; issuing it shows which instruction it encodes.
extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    constexpr auto set_all  = hal::math_counters.set<Counters::All, 0>();
    constexpr auto inc_srcb = hal::math_counters.increment<Counters::SrcB, 15>();
    constexpr auto set_bias = hal::math_counters.set<Counters::Bias, 5>();

    TTI_INSN(set_all.get_operation());
    TTI_INSN(inc_srcb.get_operation());
    TTI_INSN(set_bias.get_operation());
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttsetrwc 0,0,0,0,0,7
// CHECK-NEXT: ttincrwc 0,0,15,0
// CHECK-NEXT: ttsetibrwc 0,5,0
// CHECK-NEXT: ret
