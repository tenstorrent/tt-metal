// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --enable-var-scope
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --check-prefix=ASSERT

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

using Template1Config = mop::MopConfig<mop::MopTemplate::Template1>;

// MopCfg word i is stored at 0xffb80000 + 4 * i, after the mop_sync handshake.
// Distinct operation words with zero low bits materialize as one lui each; an
// omitted operation resolves to TT_OP_NOP (0x02000000) unless noted.
inline constexpr std::uint32_t OP_START       = 0x10001000;
inline constexpr std::uint32_t OP_END         = 0x10002000;
inline constexpr std::uint32_t OP_SECOND_END  = 0x10003000;
inline constexpr std::uint32_t OP_BODY        = 0x10004000;
inline constexpr std::uint32_t OP_ALTERNATING = 0x10005000;
inline constexpr std::uint32_t OP_LAST_FINAL  = 0x10006000;
inline constexpr std::uint32_t OP_LAST_OTHER  = 0x10007000;

inline constexpr Template1Config complete {
    .outer_loop = {.count = 4, .start_op = OP_START, .end_op = OP_END, .second_end_op = OP_SECOND_END},
    .inner_loop =
        {.count                               = 8,
         .body_op                             = OP_BODY,
         .alternating_op                      = OP_ALTERNATING,
         .last_op_on_final_outer_iteration    = OP_LAST_FINAL,
         .last_op_on_nonfinal_outer_iteration = OP_LAST_OTHER},
};

extern "C" __attribute__((noinline, used)) void program_complete()
{
    mop::program<complete>();
}

// CHECK-LABEL: <program_complete>:
// CHECK: sw [[Z:a[0-7]]],0([[SYNC:a[0-7]]])
// CHECK-NEXT: lw [[DONE:a[0-7]]],0([[SYNC]])
// CHECK-NEXT: and zero,zero,[[DONE]]
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[V:a[0-7]]],4
// CHECK-NEXT: sw [[V]],0([[CFG]])
// CHECK-NEXT: li [[V:a[0-7]]],8
// CHECK-NEXT: sw [[V]],4([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10001
// CHECK-NEXT: sw [[V]],8([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10002
// CHECK-NEXT: sw [[V]],12([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10003
// CHECK-NEXT: sw [[V]],16([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10004
// CHECK-NEXT: sw [[V]],20([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10005
// CHECK-NEXT: sw [[V]],24([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10006
// CHECK-NEXT: sw [[V]],28([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10007
// CHECK-NEXT: sw [[V]],32([[CFG]])
// CHECK-NEXT: ret

inline constexpr Template1Config body_only {
    .outer_loop = {.count = 2, .second_end_op = OP_SECOND_END},
    .inner_loop = {.count = 3, .body_op = OP_BODY},
};

extern "C" __attribute__((noinline, used)) void program_defaults()
{
    mop::program<body_only>();
}

// The second end op is dropped without an end op, and both last ops default to the body op.
// CHECK-LABEL: <program_defaults>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[V:a[0-7]]],2
// CHECK-NEXT: sw [[V]],0([[CFG]])
// CHECK-NEXT: li [[V:a[0-7]]],3
// CHECK-NEXT: sw [[V]],4([[CFG]])
// CHECK-NEXT: lui [[NOP:a[0-7]]],0x2000
// CHECK-NEXT: sw [[NOP]],8([[CFG]])
// CHECK-NEXT: sw [[NOP]],12([[CFG]])
// CHECK-NEXT: sw [[NOP]],16([[CFG]])
// CHECK-NEXT: lui [[BODY:a[0-7]]],0x10004
// CHECK-NEXT: sw [[BODY]],20([[CFG]])
// CHECK-NEXT: sw [[NOP]],24([[CFG]])
// CHECK-NEXT: sw [[BODY]],28([[CFG]])
// CHECK-NEXT: sw [[BODY]],32([[CFG]])
// CHECK-NEXT: ret

inline constexpr Template1Config alternating {
    .outer_loop = {.count = 1, .start_op = OP_START},
    .inner_loop = {.count = 2, .body_op = OP_BODY, .alternating_op = OP_ALTERNATING},
};

extern "C" __attribute__((noinline, used)) void program_alternating_defaults()
{
    mop::program<alternating>();
}

// With an alternating op, both last ops default to it instead of the body op.
// CHECK-LABEL: <program_alternating_defaults>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[V:a[0-7]]],1
// CHECK-NEXT: sw [[V]],0([[CFG]])
// CHECK-NEXT: li [[V:a[0-7]]],2
// CHECK-NEXT: sw [[V]],4([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10001
// CHECK-NEXT: sw [[V]],8([[CFG]])
// CHECK-NEXT: lui [[NOP:a[0-7]]],0x2000
// CHECK-NEXT: sw [[NOP]],12([[CFG]])
// CHECK-NEXT: sw [[NOP]],16([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10004
// CHECK-NEXT: sw [[V]],20([[CFG]])
// CHECK-NEXT: lui [[ALT:a[0-7]]],0x10005
// CHECK-NEXT: sw [[ALT]],24([[CFG]])
// CHECK-NEXT: sw [[ALT]],28([[CFG]])
// CHECK-NEXT: sw [[ALT]],32([[CFG]])
// CHECK-NEXT: ret

inline constexpr Template1Config no_start_active_end {
    .outer_loop = {.count = 4, .end_op = OP_END},
    .inner_loop = {.count = 1, .body_op = OP_BODY},
};

extern "C" __attribute__((noinline, used)) void program_runtime_counts(std::uint32_t outer, std::uint32_t inner)
{
    mop::program<complete>(mop::runtime_field<mop::Template1Field::InnerLoopCount>(inner), mop::runtime_field<mop::Template1Field::OuterLoopCount>(outer));
}

// Runtime counts replace words 0 and 1 regardless of argument order; a start op rules out the count anomaly.
// CHECK-LABEL: <program_runtime_counts>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: sw a0,0([[CFG]])
// CHECK-NEXT: sw a1,4([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10001
// CHECK-NEXT: sw [[V]],8([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10002
// CHECK-NEXT: sw [[V]],12([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10003
// CHECK-NEXT: sw [[V]],16([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10004
// CHECK-NEXT: sw [[V]],20([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10005
// CHECK-NEXT: sw [[V]],24([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10006
// CHECK-NEXT: sw [[V]],28([[CFG]])
// CHECK-NEXT: lui [[V:a[0-7]]],0x10007
// CHECK-NEXT: sw [[V]],32([[CFG]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void program_runtime_inner_count(std::uint32_t inner)
{
    mop::program<no_start_active_end>(mop::runtime_field<mop::Template1Field::InnerLoopCount>(inner));
}

// Without a start op and with an active end op, a runtime inner count whose low
// ten bits are zero triggers the count anomaly and is asserted before the sync.
// CHECK-LABEL: <program_runtime_inner_count>:
// CHECK-NEXT: andi [[LOW:a[0-7]]],a0,1023
// CHECK-NEXT: beqz [[LOW]],{{[0-9a-f]+}} <[[ANOMALY:\.L[0-9]+]]>
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[V:a[0-7]]],4
// CHECK-NEXT: sw [[V]],0([[CFG]])
// CHECK-NEXT: sw a0,4([[CFG]])
// CHECK: ret
// CHECK-EMPTY:
// CHECK-NEXT: {{^}}{{[0-9a-f]+}} <[[ANOMALY]]>:
// CHECK-NEXT: ebreak

extern "C" __attribute__((noinline, used)) void program_runtime_config(std::uint32_t inner)
{
    mop::program(Template1Config {
        .outer_loop = {.count = 4, .end_op = OP_END},
        .inner_loop = {.count = inner, .body_op = OP_BODY},
    });
}

// CHECK-LABEL: <program_runtime_config>:
// CHECK-NEXT: andi [[LOW:a[0-7]]],a0,1023
// CHECK-NEXT: beqz [[LOW]],{{[0-9a-f]+}} <[[ANOMALY:\.L[0-9]+]]>
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: li [[V:a[0-7]]],4
// CHECK-NEXT: sw [[V]],0([[CFG]])
// CHECK-NEXT: sw a0,4([[CFG]])
// CHECK: ret
// CHECK-EMPTY:
// CHECK-NEXT: {{^}}{{[0-9a-f]+}} <[[ANOMALY]]>:
// CHECK-NEXT: ebreak

// Only the two configurations that can reach the count anomaly assert.
// ASSERT-LABEL: <program_complete>:
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <program_defaults>:
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <program_alternating_defaults>:
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <program_runtime_counts>:
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <program_runtime_inner_count>:
// ASSERT-COUNT-1: ebreak
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <program_runtime_config>:
// ASSERT-COUNT-1: ebreak
// ASSERT-NOT: ebreak
// ASSERT-LABEL: <write_inner_loop_count>:
// ASSERT-NOT: ebreak

extern "C" __attribute__((noinline, used)) void write_inner_loop_count(std::uint32_t count)
{
    mop::write<mop::Template1Field::InnerLoopCount>(count);
}

// CHECK-LABEL: <write_inner_loop_count>:
// CHECK: and zero,zero,{{a[0-7]}}
// CHECK-NEXT: lui [[CFG:a[0-7]]],0xffb80
// CHECK-NEXT: sw a0,4([[CFG]])
// CHECK-NEXT: ret
