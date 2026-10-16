// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -dr %t.assert.o | FileCheck %s --check-prefix=ASSERT
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.release.o
// RUN: %{blackhole_objdump} -drz %t.release.o > %t.release.dump
// RUN: %{blackhole_compare_codegen} %t.release.dump

// Release batches must match the raw instructions, without live-state reads or
// assertion overhead. Compare adjacent functions with identical-code folding disabled.

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

extern "C" __attribute__((noinline, used)) void reference_batch_pop()
{
    TTI_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, 1 /*IncrVal*/, 3 /*WrapVal*/, 0 /*Sel32b*/, 4 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}

// ASSERT-LABEL: <reference_batch_pop>:

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

extern "C" __attribute__((noinline, used)) void reference_batch_push()
{
    TTI_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, 1 /*IncrVal*/, 3 /*WrapVal*/, 1 /*Sel32b*/, 4 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}

// ASSERT-LABEL: <reference_batch_push>:

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

extern "C" __attribute__((noinline, used)) void reference_batch_aliased_address()
{
    TTI_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, 1 /*IncrVal*/, 3 /*WrapVal*/, 0 /*Sel32b*/, 3 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}

// ASSERT-LABEL: <reference_batch_aliased_address>:

extern "C" __attribute__((noinline, used)) void runtime_batch(const std::uint8_t increment)
{
    constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
    fifo.pop(hal::gpr<4>(), increment);
}

// ASSERT-LABEL: <runtime_batch>:
// ASSERT: regfileE
// ASSERT: fence
// ASSERT: ebreak

extern "C" __attribute__((noinline, used)) void reference_runtime_batch(const std::uint8_t increment)
{
    TT_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, increment, 3 /*WrapVal*/, 0 /*Sel32b*/, 4 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}

// ASSERT-LABEL: <reference_runtime_batch>:
