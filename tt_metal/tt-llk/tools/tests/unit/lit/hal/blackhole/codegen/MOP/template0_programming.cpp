// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --enable-var-scope
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --check-prefix=ASSERT

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

using Template0Config = mop::MopConfig<mop::MopTemplate::Template0>;

// MopCfg word i is stored at 0xffb80000 + 4 * i. Every write follows the
// mop_sync store/load/and handshake. Distinct operation words with zero low
// bits materialize as one lui each; an omitted operation is TT_OP_NOP (0x02000000).
inline constexpr std::uint32_t OP_START        = 0x10001000;
inline constexpr std::uint32_t OP_MID_A        = 0x10002000;
inline constexpr std::uint32_t OP_MID_B        = 0x10003000;
inline constexpr std::uint32_t OP_MID_C        = 0x10004000;
inline constexpr std::uint32_t OP_END          = 0x10005000;
inline constexpr std::uint32_t OP_START_SHADOW = 0x10006000;
inline constexpr std::uint32_t OP_END_SHADOW   = 0x10007000;

inline constexpr Template0Config all_groups {
    .start_op        = OP_START,
    .mid_ops         = {.op_a = OP_MID_A, .op_b = OP_MID_B, .op_c = OP_MID_C},
    .end_op          = OP_END,
    .start_op_shadow = OP_START_SHADOW,
    .end_op_shadow   = OP_END_SHADOW,
};

extern "C" __attribute__((noinline, used)) void program_all_groups()
{
    mop::program<all_groups>();
}

// Word 1 enables the end group (bit 0) and the middle group (bit 1).
// CHECK-LABEL: <program_all_groups>:
// CHECK: sw [[Z:a[0-7]]],0([[SYNC:a[0-7]]])
// CHECK-NEXT: lw [[DONE:a[0-7]]],0([[SYNC]])
// CHECK-NEXT: and zero,zero,[[DONE]]
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[FLAGS:a[0-7]]],3
// CHECK-NEXT: sw [[FLAGS]],4([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10005
// CHECK-NEXT: sw [[V]],8([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10001
// CHECK-NEXT: sw [[V]],12([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10002
// CHECK-NEXT: sw [[V]],16([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10003
// CHECK-NEXT: sw [[V]],20([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10004
// CHECK-NEXT: sw [[V]],24([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10006
// CHECK-NEXT: sw [[V]],28([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10007
// CHECK-NEXT: sw [[V]],32([[CFG]])
// CHECK-NEXT: ret

inline constexpr Template0Config start_only {
    .start_op        = OP_START,
    .start_op_shadow = OP_START_SHADOW,
};

extern "C" __attribute__((noinline, used)) void program_start_only()
{
    mop::program<start_only>();
}

// Both optional groups are disabled and their slots hold NOP.
// CHECK-LABEL: <program_start_only>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: sw zero,4([[CFG]])
// CHECK-NEXT: lui [[NOP:a[0-7]]],0x2000
// CHECK-NEXT: sw [[NOP]],8([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10001
// CHECK-NEXT: sw [[V]],12([[CFG]])
// CHECK-NEXT: sw [[NOP]],16([[CFG]])
// CHECK-NEXT: sw [[NOP]],20([[CFG]])
// CHECK-NEXT: sw [[NOP]],24([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10006
// CHECK-NEXT: sw [[V]],28([[CFG]])
// CHECK-NEXT: sw [[NOP]],32([[CFG]])
// CHECK-NEXT: ret

inline constexpr Template0Config end_group {
    .start_op        = OP_START,
    .end_op          = OP_END,
    .start_op_shadow = OP_START_SHADOW,
    .end_op_shadow   = OP_END_SHADOW,
};

extern "C" __attribute__((noinline, used)) void program_end_group()
{
    mop::program<end_group>();
}

// CHECK-LABEL: <program_end_group>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[FLAGS:a[0-7]]],1
// CHECK-NEXT: sw [[FLAGS]],4([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10005
// CHECK-NEXT: sw [[V]],8([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10001
// CHECK-NEXT: sw [[V]],12([[CFG]])
// CHECK-NEXT: lui [[NOP:a[0-7]]],0x2000
// CHECK-NEXT: sw [[NOP]],16([[CFG]])
// CHECK-NEXT: sw [[NOP]],20([[CFG]])
// CHECK-NEXT: sw [[NOP]],24([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10006
// CHECK-NEXT: sw [[V]],28([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10007
// CHECK-NEXT: sw [[V]],32([[CFG]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void program_runtime_operations(std::uint32_t start, std::uint32_t mid, std::uint32_t start_shadow)
{
    mop::program(Template0Config {
        .start_op        = start,
        .mid_ops         = {.op_a = mid, .op_b = mid, .op_c = mid},
        .start_op_shadow = start_shadow,
    });
}

// CHECK-LABEL: <program_runtime_operations>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[FLAGS:a[0-7]]],2
// CHECK-NEXT: sw [[FLAGS]],4([[CFG]])
// CHECK-NEXT: lui [[NOP:a[0-7]]],0x2000
// CHECK-NEXT: sw [[NOP]],8([[CFG]])
// CHECK-NEXT: sw a0,12([[CFG]])
// CHECK-NEXT: sw a1,16([[CFG]])
// CHECK-NEXT: sw a1,20([[CFG]])
// CHECK-NEXT: sw a1,24([[CFG]])
// CHECK-NEXT: sw a2,28([[CFG]])
// CHECK-NEXT: sw [[NOP]],32([[CFG]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void program_runtime_config(const Template0Config& config)
{
    mop::program(config);
}

// A configuration whose shape is known only at runtime gets all four shape
// checks; constant shapes fold them away.
// ASSERT-LABEL: <program_runtime_operations>:
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <program_runtime_config>:
// ASSERT-COUNT-4: ebreak
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <write_mid_op_b>:
// CHECK-LABEL: <program_runtime_config>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: or [[FLAGS:[a-t][0-7]]],{{[a-t][0-7]}},{{[a-t][0-7]}}
// CHECK-NEXT: sw [[FLAGS]],4([[CFG]])
// CHECK-COUNT-7: sw {{[a-t][0-7]}},{{[0-9]+}}([[CFG]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_mid_op_b(std::uint32_t operation)
{
    mop::write<mop::Template0Field::MidOpB>(operation);
}

// CHECK-LABEL: <write_mid_op_b>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: sw a0,20([[CFG]])
// CHECK-NEXT: ret
