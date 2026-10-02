// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instruction.

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

extern "C" __attribute__((noinline, used)) void init_post_get_masks()
{
    sync::semaphore::init<sync::SemaphoreMask::MathPack | sync::SemaphoreMask::PackDone, 1, 2>();
    sync::semaphore::init<sync::SemaphoreMask::All, 0, 15>();
    sync::semaphore::post<sync::SemaphoreMask::S0>();
    sync::semaphore::get<sync::SemaphoreMask::MathDone>();
}

// CHECK-LABEL: <init_post_get_masks>:
// CHECK-NEXT: ttseminit 2,1,18
// CHECK-NEXT: ttseminit 15,0,255
// CHECK-NEXT: ttsempost 1
// CHECK-NEXT: ttsemget 128
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN((sync::semaphore::init_operation<sync::SemaphoreMask::UnpackSync, 3, 4>()));
    TTI_INSN(sync::semaphore::post_operation<sync::SemaphoreMask::S6 | sync::SemaphoreMask::S7>());
    TTI_INSN(sync::semaphore::get_operation<sync::SemaphoreMask::UnpackToDest>());
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttseminit 4,3,32
// CHECK-NEXT: ttsempost 192
// CHECK-NEXT: ttsemget 4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void post_get_tensix_index()
{
    sync::semaphore::post<sync::Access::Tensix, sync::Semaphore::MathPack>();
    sync::semaphore::get<sync::Access::Tensix, sync::Semaphore::S7>();
}

// CHECK-LABEL: <post_get_tensix_index>:
// CHECK-NEXT: ttsempost 2
// CHECK-NEXT: ttsemget 128
// CHECK-NEXT: ret

// MMIO semaphores sit at pc_buf_base + 4 * (8 + index): storing 0 posts,
// storing 1 gets, and a load reads the value.

extern "C" __attribute__((noinline, used)) void post_mmio_index()
{
    sync::semaphore::post<sync::Access::MMIO, sync::Semaphore::MathPack>();
}

// CHECK-LABEL: <post_mmio_index>:
// CHECK-NEXT: lui [[BASE:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw [[PTR:a[0-7]]],0([[BASE]])
// CHECK-NEXT: R_RISCV_LO12_I _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw zero,36([[PTR]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void get_mmio_index()
{
    sync::semaphore::get<sync::Access::MMIO, sync::Semaphore::UnpackSync>();
}

// CHECK-LABEL: <get_mmio_index>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw [[R0]],0([[R0]])
// CHECK-NEXT: R_RISCV_LO12_I _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: li [[R1:a[0-7]]],1
// CHECK-NEXT: sw [[R1]],52([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint8_t read_index()
{
    return sync::semaphore::read<sync::Semaphore::MathDone>();
}

// CHECK-LABEL: <read_index>:
// CHECK-NEXT: lui [[BASE:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw [[PTR:a[0-7]]],0([[BASE]])
// CHECK-NEXT: R_RISCV_LO12_I _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw a0,60([[PTR]])
// CHECK-NEXT: zext.b a0,a0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void init_runtime(const sync::SemaphoreMask mask, const std::uint32_t initial, const std::uint32_t maximum)
{
    sync::semaphore::init(mask, initial, maximum);
}

// CHECK-LABEL: <init_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa3000
// CHECK-NEXT: slli a2,a2,0x14
// CHECK-NEXT: add a2,a2,[[R0]]
// CHECK-NEXT: slli a1,a1,0x10
// CHECK-NEXT: lui [[R0]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add a2,a2,a1
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sh2add a0,a0,a2
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void post_runtime(const sync::SemaphoreMask mask)
{
    sync::semaphore::post(mask);
}

// CHECK-LABEL: <post_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lui [[R1:a[0-7]]],0xa4000
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sh2add a0,a0,[[R1]]
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void get_runtime(const sync::SemaphoreMask mask)
{
    sync::semaphore::get(mask);
}

// CHECK-LABEL: <get_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lui [[R1:a[0-7]]],0xa5000
// CHECK-NEXT: mv [[R0]],[[R0]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sh2add a0,a0,[[R1]]
// CHECK-NEXT: sw a0,0([[R0]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void post_tensix_runtime(const sync::Semaphore semaphore)
{
    sync::semaphore::post<sync::Access::Tensix>(semaphore);
}

// CHECK-LABEL: <post_tensix_runtime>:
// CHECK-NEXT: li [[R0:a[0-7]]],4
// CHECK-NEXT: sll [[R0]],[[R0]],a0
// CHECK-NEXT: lui [[R1:a[0-7]]],0xa4000
// CHECK-NEXT: lui [[R2:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add [[R0]],[[R0]],[[R1]]
// CHECK-NEXT: mv [[R2]],[[R2]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw [[R0]],0([[R2]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void get_tensix_runtime(const sync::Semaphore semaphore)
{
    sync::semaphore::get<sync::Access::Tensix>(semaphore);
}

// CHECK-LABEL: <get_tensix_runtime>:
// CHECK-NEXT: li [[R0:a[0-7]]],4
// CHECK-NEXT: sll [[R0]],[[R0]],a0
// CHECK-NEXT: lui [[R1:a[0-7]]],0xa5000
// CHECK-NEXT: lui [[R2:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add [[R0]],[[R0]],[[R1]]
// CHECK-NEXT: mv [[R2]],[[R2]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw [[R0]],0([[R2]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void post_mmio_runtime(const sync::Semaphore semaphore)
{
    sync::semaphore::post<sync::Access::MMIO>(semaphore);
}

// CHECK-LABEL: <post_mmio_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw [[R0]],0([[R0]])
// CHECK-NEXT: R_RISCV_LO12_I _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sh2add a0,a0,[[R0]]
// CHECK-NEXT: sw zero,32(a0)
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void get_mmio_runtime(const sync::Semaphore semaphore)
{
    sync::semaphore::get<sync::Access::MMIO>(semaphore);
}

// CHECK-LABEL: <get_mmio_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw [[R1:a[0-7]]],0([[R0]])
// CHECK-NEXT: R_RISCV_LO12_I _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: li [[R0]],1
// CHECK-NEXT: sh2add a0,a0,[[R1]]
// CHECK-NEXT: sw [[R0]],32(a0)
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint8_t read_runtime(const sync::Semaphore semaphore)
{
    return sync::semaphore::read(semaphore);
}

// CHECK-LABEL: <read_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: lw [[R0]],0([[R0]])
// CHECK-NEXT: R_RISCV_LO12_I _ZN7ckernel11pc_buf_baseE
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sh2add a0,a0,[[R0]]
// CHECK-NEXT: lw a0,32(a0)
// CHECK-NEXT: zext.b a0,a0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_init_runtime(
    const sync::SemaphoreMask mask, const std::uint32_t initial, const std::uint32_t maximum)
{
    return sync::semaphore::init_operation(mask, initial, maximum);
}

// CHECK-LABEL: <encode_init_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa3000
// CHECK-NEXT: slli a2,a2,0x14
// CHECK-NEXT: slli a1,a1,0x10
// CHECK-NEXT: add a2,a2,[[R0]]
// CHECK-NEXT: add a2,a2,a1
// CHECK-NEXT: sh2add a0,a0,a2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_post_runtime(const sync::SemaphoreMask mask)
{
    return sync::semaphore::post_operation(mask);
}

// CHECK-LABEL: <encode_post_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa4000
// CHECK-NEXT: sh2add a0,a0,[[R0]]
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_get_runtime(const sync::SemaphoreMask mask)
{
    return sync::semaphore::get_operation(mask);
}

// CHECK-LABEL: <encode_get_runtime>:
// CHECK-NEXT: lui [[R0:a[0-7]]],0xa5000
// CHECK-NEXT: sh2add a0,a0,[[R0]]
// CHECK-NEXT: ret
