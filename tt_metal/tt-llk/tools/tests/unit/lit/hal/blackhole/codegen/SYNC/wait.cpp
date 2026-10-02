// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instruction.

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

extern "C" __attribute__((noinline, used)) void stall_variants()
{
    sync::wait::stall<sync::StallTarget::Config | sync::StallTarget::Sfpu, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    sync::wait::stall<sync::StallTarget::All, sync::StallCondition::All>();
    sync::wait::stall<sync::StallTarget::HardwareDefault, sync::StallCondition::HardwareDefault>();
    sync::wait::stall<sync::StallTarget::Thcon, sync::StallCondition::SrcAValid | sync::StallCondition::SrcBValid>();
}

// CHECK-LABEL: <stall_variants>:
// CHECK-NEXT: ttstallwait 384,2064
// CHECK-NEXT: ttstallwait 511,8191
// CHECK-NEXT: ttstallwait 0,0
// CHECK-NEXT: ttstallwait 32,384
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void semaphore_wait_variants()
{
    sync::wait::semaphore<sync::StallTarget::Sync | sync::StallTarget::Math, sync::SemaphoreMask::MathPack, sync::SemaphoreCondition::WhileZero>();
    sync::wait::semaphore<sync::StallTarget::Pack, sync::SemaphoreMask::S0 | sync::SemaphoreMask::S7, sync::SemaphoreCondition::WhileMaximum>();
    sync::wait::semaphore<sync::StallTarget::Unpack, sync::SemaphoreMask::All, sync::SemaphoreCondition::WhileZero | sync::SemaphoreCondition::WhileMaximum>();
}

// CHECK-LABEL: <semaphore_wait_variants>:
// CHECK-NEXT: ttsemwait 66,2,1
// CHECK-NEXT: ttsemwait 4,129,2
// CHECK-NEXT: ttsemwait 8,255,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void stream_wait_variants()
{
    sync::wait::stream<sync::StallTarget::Unpack, sync::StreamSlot::S0, sync::StreamTarget::Phase, 0>();
    sync::wait::stream<sync::StallTarget::All, sync::StreamSlot::S3, sync::StreamTarget::MessagesReceived, 1023>();
}

// CHECK-LABEL: <stream_wait_variants>:
// CHECK-NEXT: ttstreamwait 8,0,0,0
// CHECK-NEXT: ttstreamwait 511,1023,1,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN((sync::wait::stall_operation<sync::StallTarget::Pack, sync::StallCondition::PackerIdle>()));
    TTI_INSN((sync::wait::semaphore_operation<sync::StallTarget::Mover, sync::SemaphoreMask::UnpackToDest, sync::SemaphoreCondition::WhileZero>()));
    TTI_INSN((sync::wait::stream_operation<sync::StallTarget::Math, sync::StreamSlot::S1, sync::StreamTarget::Phase, 513>()));
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttstallwait 4,8
// CHECK-NEXT: ttsemwait 16,4,1
// CHECK-NEXT: ttstreamwait 64,513,0,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void stall_runtime(const sync::StallTarget targets, const sync::StallCondition conditions)
{
    sync::wait::stall(targets, conditions);
}

// CHECK-LABEL: <stall_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa2000
// CHECK-NEXT: add a1,a1,[[R0]]
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add a0,a0,a1
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void semaphore_runtime(
    const sync::StallTarget targets, const sync::SemaphoreMask mask, const sync::SemaphoreCondition conditions)
{
    sync::wait::semaphore(targets, mask, conditions);
}

// CHECK-LABEL: <semaphore_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa6000
// CHECK-NEXT: add a2,a2,[[R0]]
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: sh2add a1,a1,a2
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add a1,a1,a0
// CHECK-NEXT: sw a1,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void stream_runtime(
    const sync::StallTarget targets, const sync::StreamSlot slot, const sync::StreamTarget target, const std::uint32_t target_low)
{
    sync::wait::stream(targets, slot, target, target_low);
}

// CHECK-LABEL: <stream_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa7000
// CHECK-NEXT: slli a3,a3,0x4
// CHECK-NEXT: add a3,a3,[[R0]]
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: sh3add a2,a2,a3
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add a2,a2,a1
// CHECK-NEXT: add a2,a2,a0
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a2,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_stall_runtime(const sync::StallTarget targets, const sync::StallCondition conditions)
{
    return sync::wait::stall_operation(targets, conditions);
}

// CHECK-LABEL: <encode_stall_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa2000
// CHECK-NEXT: add a1,a1,[[R0]]
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: add a0,a0,a1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_runtime(
    const sync::StallTarget targets, const sync::SemaphoreMask mask, const sync::SemaphoreCondition conditions)
{
    return sync::wait::semaphore_operation(targets, mask, conditions);
}

// CHECK-LABEL: <encode_semaphore_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa6000
// CHECK-NEXT: add a2,a2,[[R0]]
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: sh2add a1,a1,a2
// CHECK-NEXT: add a0,a1,a0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_stream_runtime(
    const sync::StallTarget targets, const sync::StreamSlot slot, const sync::StreamTarget target, const std::uint32_t target_low)
{
    return sync::wait::stream_operation(targets, slot, target, target_low);
}

// CHECK-LABEL: <encode_stream_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa7000
// CHECK-NEXT: slli a3,a3,0x4
// CHECK-NEXT: add a3,a3,[[R0]]
// CHECK-NEXT: sh3add a2,a2,a3
// CHECK-NEXT: slli a0,a0,0xf
// CHECK-NEXT: add a2,a2,a1
// CHECK-NEXT: add a0,a2,a0
// CHECK-NEXT: ret
