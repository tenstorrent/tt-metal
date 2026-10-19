// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/counters/rwc.h"

namespace rwc = hal::counters::rwc;

extern "C" __attribute__((noinline, used)) void set_sources()
{
    constexpr auto sources = rwc::MathCounter {.src_a = rwc::Set(4), .src_b = rwc::Set(4)};
    rwc::set<sources>();
}

// CHECK-LABEL: <set_sources>:
// CHECK-NEXT: ttsetrwc 0,0,0,4,4,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_all_and_clear_fidelity()
{
    constexpr auto reset = rwc::MathCounter {.src_a = rwc::Set(1), .src_b = rwc::Set(2), .dest = rwc::Set(3), .clear_fidelity = true};
    rwc::set<reset>();
}

// CHECK-LABEL: <set_all_and_clear_fidelity>:
// CHECK-NEXT: ttsetrwc 0,0,3,2,1,15
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_dest()
{
    constexpr auto rows = rwc::MathCounter {.dest = rwc::Increment(8)};
    rwc::set<rows>();
}

// CHECK-LABEL: <increment_dest>:
// CHECK-NEXT: ttincrwc 0,8,0,0
// CHECK-NEXT: ret

// Without a SETRWC-only field, AdvanceAndReset is encoded in INCRWC.
extern "C" __attribute__((noinline, used)) void advance_and_reset_dest()
{
    constexpr auto base = rwc::MathCounter {.dest = rwc::AdvanceAndReset(8)};
    rwc::set<base>();
}

// CHECK-LABEL: <advance_and_reset_dest>:
// CHECK-NEXT: ttincrwc 4,8,0,0
// CHECK-NEXT: ret

// A zero AdvanceAndReset rewinds the live counter to its carry shadow.
extern "C" __attribute__((noinline, used)) void rewind_sources()
{
    constexpr auto rewind = rwc::MathCounter {.src_a = rwc::AdvanceAndReset(), .src_b = rwc::AdvanceAndReset()};
    rwc::set<rewind>();
}

// CHECK-LABEL: <rewind_sources>:
// CHECK-NEXT: ttincrwc 3,0,0,0
// CHECK-NEXT: ret

// Set and AdvanceAndReset share one SETRWC; the flip releases both source banks.
extern "C" __attribute__((noinline, used)) void next_face()
{
    constexpr auto face = rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::AdvanceAndReset(8), .flip = rwc::Src::A | rwc::Src::B};
    rwc::set<face>();
}

// CHECK-LABEL: <next_face>:
// CHECK-NEXT: ttsetrwc 3,4,8,0,0,6
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void flip_only()
{
    constexpr auto flip = rwc::MathCounter {.flip = rwc::Src::B};
    rwc::set<flip>();
}

// CHECK-LABEL: <flip_only>:
// CHECK-NEXT: ttsetrwc 2,0,0,0,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_and_save_dest()
{
    constexpr auto save = rwc::MathCounter {.dest = rwc::IncrementAndSave(7)};
    rwc::set<save>();
}

// CHECK-LABEL: <increment_and_save_dest>:
// CHECK-NEXT: ttsetrwc 0,8,7,0,0,4
// CHECK-NEXT: ret

// Set and Increment need both instructions; SETRWC is issued first.
extern "C" __attribute__((noinline, used)) void set_and_increment()
{
    constexpr auto mixed = rwc::MathCounter {.src_a = rwc::Set(1), .dest = rwc::Increment(2)};
    rwc::set<mixed>();
}

// CHECK-LABEL: <set_and_increment>:
// CHECK-NEXT: ttsetrwc 0,0,0,0,1,1
// CHECK-NEXT: ttincrwc 0,2,0,0
// CHECK-NEXT: ret

// operation() returns the encoded word; issuing it shows which instruction it encodes.
extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    constexpr auto set_all  = rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0)}.operation();
    constexpr auto inc_srcb = rwc::MathCounter {.src_b = rwc::Increment(15)}.operation();

    TTI_INSN(set_all);
    TTI_INSN(inc_srcb);
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttsetrwc 0,0,0,0,0,7
// CHECK-NEXT: ttincrwc 0,0,15,0
// CHECK-NEXT: ret
