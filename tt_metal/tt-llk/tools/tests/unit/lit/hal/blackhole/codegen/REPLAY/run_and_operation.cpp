// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/mop.h"
#include "hal/replay.h"

namespace mop    = hal::mop;
namespace replay = hal::replay;

// REPLAY operands are (start, count, execute while recording, record); the
// encoded word is 0x04000000 | start << 14 | (count & 63) << 4.

extern "C" __attribute__((noinline, used)) void run_full_buffer()
{
    replay::run<replay::BufferRange {0, 32}>();
}

// CHECK-LABEL: <run_full_buffer>:
// CHECK-NEXT: ttreplay 0,32,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_wrapping()
{
    replay::run<replay::BufferRange {31, 2}>();
}

// CHECK-LABEL: <run_wrapping>:
// CHECK-NEXT: ttreplay 31,2,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_constant_runtime_range()
{
    replay::run(replay::BufferRange {5, 3});
}

// CHECK-LABEL: <run_constant_runtime_range>:
// CHECK-NEXT: ttreplay 5,3,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void run_runtime_count(std::uint32_t count)
{
    replay::run(replay::BufferRange {5, count});
}

// A count of 64 is encoded as zero by the low six bits.
// CHECK-LABEL: <run_runtime_count>:
// CHECK-NEXT: addi [[ENCODED:a[0-7]]],a0,-1
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bgeu [[LIMIT]],[[ENCODED]],
// CHECK: ebreak
// CHECK: andi [[COUNT:a[0-7]]],a0,63
// CHECK-NEXT: lui [[OP:a[0-7]]],0x4014
// CHECK-NEXT: slli [[COUNT]],[[COUNT]],0x4
// CHECK-NEXT: add [[WORD:a[0-7]]],[[COUNT]],[[OP]]
// CHECK-NEXT: lui [[BUF:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[WORD]],0([[BUF]])
// CHECK-NOT: ttreplay
// CHECK: ret

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_max_count()
{
    return replay::get_operation<replay::BufferRange {4, 64}>();
}

// CHECK-LABEL: <get_operation_max_count>:
// CHECK-NEXT: lui a0,0x4010
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_last_slot()
{
    return replay::get_operation<replay::BufferRange {31, 32}>();
}

// 0x0407c200
// CHECK-LABEL: <get_operation_last_slot>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0x407c
// CHECK-NEXT: addi a0,[[OP]],512
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_runtime(std::uint32_t start, std::uint32_t count)
{
    return replay::get_operation({start, count});
}

// Both fields may be runtime values; the start must be below 32 and the count in [1, 64].
// CHECK-LABEL: <get_operation_runtime>:
// CHECK-NEXT: addi [[ENCODED:a[0-7]]],a1,-1
// CHECK-NEXT: sltiu [[COUNT_OK:a[0-7]]],[[ENCODED]],64
// CHECK-NEXT: sltiu [[START_OK:a[0-7]]],a0,32
// CHECK-NEXT: and [[OK:a[0-7]]],[[COUNT_OK]],[[START_OK]]
// CHECK-NEXT: bnez [[OK]],
// CHECK: ebreak
// CHECK: slli [[COUNT:a[0-7]]],a1,0x4
// CHECK-NEXT: slli [[START:a[0-7]]],a0,0xe
// CHECK-NEXT: andi [[COUNT]],[[COUNT]],1008
// CHECK-NEXT: add [[FIELDS:a[0-7]]],[[COUNT]],[[START]]
// CHECK-NEXT: lui [[OP:a[0-7]]],0x4000
// CHECK-NEXT: add a0,[[FIELDS]],[[OP]]
// CHECK-NEXT: ret

inline constexpr replay::BufferRange row_epilogue {16, 2};

inline constexpr mop::MopConfig<mop::MopTemplate::Template1> replay_body {
    .outer_loop = {.count = 1, .start_op = TT_OP_NOP},
    .inner_loop = {.count = 4, .body_op = replay::get_operation<row_epilogue>()},
};

extern "C" __attribute__((noinline, used)) void program_mop_with_replay()
{
    mop::program<replay_body>();
}

// The playback word 0x04040020 lands in the Template 1 body and both last-op slots.
// CHECK-LABEL: <program_mop_with_replay>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[V:a[0-7]]],1
// CHECK-NEXT: sw [[V]],0([[CFG]])
// CHECK-NEXT: li [[V:a[0-7]]],4
// CHECK-NEXT: sw [[V]],4([[CFG]])
// CHECK-NEXT: lui [[NOP:a[0-7]]],0x2000
// CHECK-NEXT: sw [[NOP]],8([[CFG]])
// CHECK-NEXT: sw [[NOP]],12([[CFG]])
// CHECK-NEXT: lui [[OP:a[0-7]]],0x4040
// CHECK-NEXT: sw [[NOP]],16([[CFG]])
// CHECK-NEXT: addi [[OP]],[[OP]],32
// CHECK-NEXT: sw [[OP]],20([[CFG]])
// CHECK-NEXT: sw [[NOP]],24([[CFG]])
// CHECK-NEXT: sw [[OP]],28([[CFG]])
// CHECK-NEXT: sw [[OP]],32([[CFG]])
// CHECK-NEXT: ret
