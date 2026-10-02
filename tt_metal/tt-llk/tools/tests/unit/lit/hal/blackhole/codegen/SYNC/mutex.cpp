// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// LLK_ASSERT is disabled so runtime paths contain only the issued instruction.

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

extern "C" __attribute__((noinline, used)) void acquire_release_physical()
{
    sync::mutex::acquire<sync::Mutex::M0>();
    sync::mutex::acquire<sync::Mutex::M2>();
    sync::mutex::release<sync::Mutex::M3>();
    sync::mutex::release<sync::Mutex::M4>();
}

// CHECK-LABEL: <acquire_release_physical>:
// CHECK-NEXT: ttatgetm 0
// CHECK-NEXT: ttatgetm 2
// CHECK-NEXT: ttatrelm 3
// CHECK-NEXT: ttatrelm 4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void acquire_release_aliases()
{
    sync::mutex::acquire<sync::Mutex::RegisterRmw>();
    sync::mutex::acquire<sync::Mutex::Unpacker0>();
    sync::mutex::acquire<sync::Mutex::Unpacker1>();
    sync::mutex::release<sync::Mutex::Sfpu>();
}

// CHECK-LABEL: <acquire_release_aliases>:
// CHECK-NEXT: ttatgetm 0
// CHECK-NEXT: ttatgetm 2
// CHECK-NEXT: ttatgetm 3
// CHECK-NEXT: ttatrelm 4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN(sync::mutex::acquire_operation<sync::Mutex::Packer0>());
    TTI_INSN(sync::mutex::release_operation<sync::Mutex::Math>());
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttatgetm 4
// CHECK-NEXT: ttatrelm 0
// CHECK-NEXT: ret

// Runtime forms add the mutex index to the opcode word (ATGETM 0xa0, ATRELM 0xa1)
// and push it through the instruction buffer.

extern "C" __attribute__((noinline, used)) void acquire_runtime(const sync::Mutex mutex)
{
    sync::mutex::acquire(mutex);
}

// CHECK-LABEL: <acquire_runtime>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0xa0000
// CHECK-NEXT: lui [[BUF:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add a0,a0,[[OP]]
// CHECK-NEXT: mv [[BUF]],[[BUF]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a0,0([[BUF]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void release_runtime(const sync::Mutex mutex)
{
    sync::mutex::release(mutex);
}

// CHECK-LABEL: <release_runtime>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0xa1000
// CHECK-NEXT: lui [[BUF:a[0-7]]],0x0
// CHECK-NEXT: R_RISCV_HI20 __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: add a0,a0,[[OP]]
// CHECK-NEXT: mv [[BUF]],[[BUF]]
// CHECK-NEXT: R_RISCV_LO12_I __instrn_buffer
// CHECK-NEXT: R_RISCV_RELAX
// CHECK-NEXT: sw a0,0([[BUF]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_acquire_runtime(const sync::Mutex mutex)
{
    return sync::mutex::acquire_operation(mutex);
}

// CHECK-LABEL: <encode_acquire_runtime>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0xa0000
// CHECK-NEXT: add a0,a0,[[OP]]
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t encode_release_runtime(const sync::Mutex mutex)
{
    return sync::mutex::release_operation(mutex);
}

// CHECK-LABEL: <encode_release_runtime>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0xa1000
// CHECK-NEXT: add a0,a0,[[OP]]
// CHECK-NEXT: ret
