// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/replay.h"

namespace replay = hal::replay;

// REPLAY operands are (start, count, execute while recording, record).

extern "C" __attribute__((noinline, used)) void record_only()
{
    replay::record<replay::BufferRange {16, 2}>(
        []
        {
            TTI_NOP;
            TTI_DMANOP;
        });
}

// CHECK-LABEL: <record_only>:
// CHECK-NEXT: ttreplay 16,2,0,1
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttdmanop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void record_and_execute_wrapping()
{
    replay::record<replay::BufferRange {30, 4}, replay::RecordBehavior::RecordAndExecute>(
        []
        {
            TTI_NOP;
            TTI_DMANOP;
            TTI_NOP;
            TTI_DMANOP;
        });
}

// CHECK-LABEL: <record_and_execute_wrapping>:
// CHECK-NEXT: ttreplay 30,4,1,1
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttdmanop
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttdmanop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void record_forwarded_arguments()
{
    replay::record<replay::BufferRange {0, 3}>(
        [](const std::uint32_t count)
        {
            for (std::uint32_t i = 0; i < count; ++i)
            {
                TTI_NOP;
            }
        },
        3);
}

// CHECK-LABEL: <record_forwarded_arguments>:
// CHECK-NEXT: ttreplay 0,3,0,1
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttnop
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void record_runtime_count(std::uint32_t count)
{
    replay::record(
        replay::BufferRange {8, count},
        [](const std::uint32_t n)
        {
            for (std::uint32_t i = 0; i < n; ++i)
            {
                TTI_NOP;
            }
        },
        count);
}

// The REPLAY word 0x04020001 | (count & 63) << 4 goes through the instruction
// buffer before the recorded body; counts outside [1, 64] assert.
// CHECK-LABEL: <record_runtime_count>:
// CHECK-NEXT: addi [[ENCODED:a[0-7]]],a0,-1
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: lui [[OP:a[0-7]]],0x4020
// CHECK-NEXT: addi [[OP]],[[OP]],1
// CHECK-NEXT: bgeu [[LIMIT]],[[ENCODED]],
// CHECK: ebreak
// CHECK-NOT: ttnop
// CHECK: andi [[COUNT:a[0-7]]],{{a[0-7]}},63
// CHECK-NEXT: slli [[COUNT]],[[COUNT]],0x4
// CHECK-NEXT: add [[WORD:a[0-7]]],[[COUNT]],[[OP]]
// CHECK-NEXT: lui [[BUF:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0([[BUF]])
// CHECK-NOT: ttreplay
// CHECK: ttnop

extern "C" __attribute__((noinline, used)) void record_and_execute_runtime_count(std::uint32_t count)
{
    replay::record<replay::RecordBehavior::RecordAndExecute>(
        replay::BufferRange {8, count},
        [](const std::uint32_t n)
        {
            for (std::uint32_t i = 0; i < n; ++i)
            {
                TTI_NOP;
            }
        },
        count);
}

// CHECK-LABEL: <record_and_execute_runtime_count>:
// CHECK-NEXT: addi [[ENCODED:a[0-7]]],a0,-1
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: lui [[OP:a[0-7]]],0x4020
// CHECK-NEXT: addi [[OP]],[[OP]],3
// CHECK-NEXT: bgeu [[LIMIT]],[[ENCODED]],
// CHECK: ebreak
// CHECK-NOT: ttnop
// CHECK: add [[WORD:a[0-7]]],{{a[0-7]}},[[OP]]
// CHECK: sw [[WORD]],0({{a[0-7]}})
// CHECK-NOT: ttreplay
// CHECK: ttnop

extern "C" __attribute__((noinline, used)) void record_constant_runtime_range()
{
    replay::record(replay::BufferRange {12, 1}, [] { TTI_NOP; });
}

// A constant range passed to the runtime overload still lowers to an immediate REPLAY.
// CHECK-LABEL: <record_constant_runtime_range>:
// CHECK-NEXT: ttreplay 12,1,0,1
// CHECK-NEXT: ttnop
// CHECK-NEXT: ret
