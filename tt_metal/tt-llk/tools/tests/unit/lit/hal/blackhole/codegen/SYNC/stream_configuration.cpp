// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o > %t.dis
// RUN: FileCheck %s --enable-var-scope < %t.dis
// RUN: FileCheck %s --check-prefix=BRANCH < %t.dis

// LLK_ASSERT is disabled so runtime paths contain only the issued instructions.
// STREAM_ID_SYNC slots S0-S3 are thread CFG words 59-62 (stream ID = group << 3 | number);
// STREAMWAIT_PHASE_HI and STREAMWAIT_NUM_MSGS_HI are words 57 and 58 and take target >> 10.

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

extern "C" __attribute__((noinline, used)) void configure_each_slot()
{
    sync::wait::configure_stream<sync::StreamSlot::S0, 0, 1>();
    sync::wait::configure_stream<sync::StreamSlot::S1, 1, 0>();
    sync::wait::configure_stream<sync::StreamSlot::S2, 7, 7>();
    sync::wait::configure_stream<sync::StreamSlot::S3, 2, 5>();
}

// CHECK-LABEL: <configure_each_slot>:
// CHECK-NEXT: ttsetc16 59,1
// CHECK-NEXT: ttsetc16 60,8
// CHECK-NEXT: ttsetc16 61,63
// CHECK-NEXT: ttsetc16 62,21
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_slot_with_stream_id(const sync::StreamId stream_id)
{
    sync::wait::configure_stream<sync::StreamSlot::S2>(stream_id);
}

// CHECK-LABEL: <configure_slot_with_stream_id>:
// CHECK-NEXT: zext.b [[R0:a[0-7]]],a0
// CHECK-NEXT: srli a0,a0,0x8
// CHECK-NEXT: slli [[R0]],[[R0]],0x3
// CHECK-NEXT: zext.b a0,a0
// CHECK-NEXT: or [[R0]],[[R0]],a0
// CHECK-NEXT: lui [[R1:a[0-7]]],0xb23d0
// CHECK-NEXT: lui [[R2:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add [[R0]],[[R0]],[[R1]]
// CHECK-NEXT: mv [[R2]],[[R2]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: addi sp,sp,-16
// CHECK-NEXT: sw [[R0]],0([[R2]])
// CHECK-NEXT: addi sp,sp,16
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_targets()
{
    sync::wait::configure_stream_target<sync::StreamTarget::Phase, 0xfffff>();
    sync::wait::configure_stream_target<sync::StreamTarget::MessagesReceived, 0x1ffff>();
    sync::wait::configure_stream_target<sync::StreamTarget::Phase, 1023>();
}

// CHECK-LABEL: <configure_targets>:
// CHECK-NEXT: ttsetc16 57,1023
// CHECK-NEXT: ttsetc16 58,127
// CHECK-NEXT: ttsetc16 57,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_and_wait_phase()
{
    sync::wait::configure_and_wait_stream<sync::StallTarget::Unpack, sync::StreamSlot::S3, 5, 6, sync::StreamTarget::Phase, 0x12345>();
}

// The target-high write must land before STREAMWAIT reads it, hence the config-idle stall.
// CHECK-LABEL: <configure_and_wait_phase>:
// CHECK-NEXT: ttsetc16 62,46
// CHECK-NEXT: ttsetc16 57,72
// CHECK-NEXT: ttstallwait 2,4096
// CHECK-NEXT: ttstreamwait 8,837,0,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_and_wait_messages(const sync::StreamId stream_id)
{
    sync::wait::
        configure_and_wait_stream<sync::StallTarget::Math | sync::StallTarget::Pack, sync::StreamSlot::S1, sync::StreamTarget::MessagesReceived, 0x1ffff>(
            stream_id);
}

// CHECK-LABEL: <configure_and_wait_messages>:
// CHECK-NEXT: zext.b [[R0:a[0-7]]],a0
// CHECK-NEXT: srli a0,a0,0x8
// CHECK-NEXT: zext.b a0,a0
// CHECK-NEXT: slli [[R0]],[[R0]],0x3
// CHECK-NEXT: or [[R0]],[[R0]],a0
// CHECK-NEXT: lui [[R1:a[0-7]]],0xb23c0
// CHECK-NEXT: lui [[R2:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add [[R0]],[[R0]],[[R1]]
// CHECK-NEXT: mv [[R2]],[[R2]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: addi sp,sp,-16
// CHECK-NEXT: sw [[R0]],0([[R2]])
// CHECK-NEXT: ttsetc16 58,127
// CHECK-NEXT: ttstallwait 2,4096
// CHECK-NEXT: ttstreamwait 68,1023,1,1
// CHECK-NEXT: addi sp,sp,16
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_stream_runtime(const sync::StreamSlot slot, const sync::StreamId stream_id)
{
    sync::wait::configure_stream(slot, stream_id);
}

// Each runtime slot dispatches to the SETC16 of its own STREAM_ID_SYNC word.
// BRANCH-LABEL: <configure_stream_runtime>:
// BRANCH-DAG: slli {{a[0-7]}},{{a[0-7]}},0x3
// BRANCH-DAG: lui {{a[0-7]}},0xb23b0
// BRANCH-DAG: lui {{a[0-7]}},0xb23c0
// BRANCH-DAG: lui {{a[0-7]}},0xb23d0
// BRANCH-DAG: lui {{a[0-7]}},0xb23e0

extern "C" __attribute__((noinline, used)) void configure_stream_target_runtime(const sync::StreamTarget target, const std::uint32_t full_target)
{
    sync::wait::configure_stream_target(target, full_target);
}

// CHECK-LABEL: <configure_stream_target_runtime>:
// CHECK-NEXT: srli a1,a1,0xa
// CHECK-NEXT: zext.h a1,a1
// CHECK-NEXT: bnez a0,
// CHECK-NEXT: R_RISCV_BRANCH
// CHECK-NEXT: lui [[PHASE:a[0-7]]],0xb2390
// CHECK: add a1,a1,[[PHASE]]
// CHECK: sw a1,0(
// CHECK-NEXT: ret
// CHECK: lui [[MESSAGES:a[0-7]]],0xb23a0
// CHECK: add a1,a1,[[MESSAGES]]
// CHECK: sw a1,0(
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_and_wait_runtime(
    const sync::StallTarget targets,
    const sync::StreamSlot slot,
    const sync::StreamId stream_id,
    const sync::StreamTarget target,
    const std::uint32_t full_target)
{
    sync::wait::configure_and_wait_stream(targets, slot, stream_id, target, full_target);
}

// All slot and target-high writes are reachable; every path then joins the
// config-idle STALLWAIT followed by STREAMWAIT with the low ten target bits.
// BRANCH-LABEL: <configure_and_wait_runtime>:
// BRANCH-DAG: lui {{a[0-7]}},0xb23b0
// BRANCH-DAG: lui {{a[0-7]}},0xb23c0
// BRANCH-DAG: lui {{a[0-7]}},0xb23d0
// BRANCH-DAG: lui {{a[0-7]}},0xb23e0
// BRANCH-DAG: lui {{a[0-7]}},0xb2390
// BRANCH-DAG: lui {{a[0-7]}},0xb23a0

// CHECK-LABEL: <configure_and_wait_runtime>:
// CHECK: andi a4,a4,1023
// CHECK-NEXT: lui [[STREAM:a[0-7]]],0xa7000
// CHECK-NEXT: slli a4,a4,0x4
// CHECK-NEXT: add a4,a4,[[STREAM]]
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: sh3add a3,a3,a4
// CHECK-NEXT: lui [[STALL:a[0-7]]],0xa2011
// CHECK-NEXT: add a3,a3,a1
// CHECK-NEXT: sw [[STALL]],0([[BUF:a[0-7]]])
// CHECK-NEXT: add a3,a3,a0
// CHECK-NEXT: sw a3,0([[BUF]])
// CHECK-NEXT: addi sp,sp,16
// CHECK-NEXT: ret
