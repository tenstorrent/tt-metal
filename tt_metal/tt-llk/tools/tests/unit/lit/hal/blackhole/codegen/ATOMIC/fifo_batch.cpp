// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -dr %t.assert.o | FileCheck %s --check-prefix=ASSERT
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.release.o
// RUN: %{blackhole_objdump} -d %t.release.o | FileCheck %s --check-prefix=RELEASE

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

extern "C" __attribute__((noinline, used)) void batch_pop()
{
    constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
    fifo.pop<1>(hal::gpr<4>());
}

// ASSERT-LABEL: <batch_pop>:
// ASSERT: ttstallwait 32,1
// ASSERT: pc_buf_baseE
// ASSERT: regfileE
// ASSERT: fence
// ASSERT-DAG: ttatincgetptr 0,0,1,3,0,4,3
// ASSERT-DAG: ebreak

// RELEASE-LABEL: <batch_pop>:
// RELEASE-NEXT: ttatincgetptr 0,0,1,3,0,4,3
// RELEASE-NEXT: ret

extern "C" __attribute__((noinline, used)) void batch_push()
{
    constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
    fifo.push<1>(hal::gpr<4>());
}

// ASSERT-LABEL: <batch_push>:
// ASSERT: ttstallwait 32,1
// ASSERT: pc_buf_baseE
// ASSERT: regfileE
// ASSERT: fence
// ASSERT-DAG: ttatincgetptr 0,0,1,3,1,4,3
// ASSERT-DAG: ebreak

// RELEASE-LABEL: <batch_push>:
// RELEASE-NEXT: ttatincgetptr 0,0,1,3,1,4,3
// RELEASE-NEXT: ret

// The assertion reads the address before the update can overwrite the aliased GPR.
extern "C" __attribute__((noinline, used)) void batch_aliased_address()
{
    constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
    fifo.pop<1>(hal::gpr<3>());
}

// ASSERT-LABEL: <batch_aliased_address>:
// ASSERT: regfileE
// ASSERT: fence
// ASSERT-NOT: ttsetdmareg
// ASSERT: ttatincgetptr 0,0,1,3,0,3,3

// RELEASE-LABEL: <batch_aliased_address>:
// RELEASE-NEXT: ttatincgetptr 0,0,1,3,0,3,3
// RELEASE-NEXT: ret

extern "C" __attribute__((noinline, used)) void runtime_batch(const std::uint8_t increment)
{
    constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
    fifo.pop(hal::gpr<4>(), increment);
}

// ASSERT-LABEL: <runtime_batch>:
// ASSERT: regfileE
// ASSERT: fence
// ASSERT: ebreak

// RELEASE-LABEL: <runtime_batch>:
// RELEASE-NOT: fence
// RELEASE-NOT: ebreak
// RELEASE: sw
// RELEASE-NEXT: ret
