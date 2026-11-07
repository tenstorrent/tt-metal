// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// codegen.cpp is compiled without LLK_ASSERT so runtime paths contain only the
// issued instruction; asserts.cpp checks the runtime operand asserts.
// Semaphore selectors are one bit per semaphore: S<n> is bit n.

//--- common.h
#pragma once

#include "hal/sync.h"

namespace sync = hal::sync;

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void stall_variants()
{
    sync::wait::stall<sync::StallTarget::Config | sync::StallTarget::Sfpu, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    sync::wait::stall<sync::StallTarget::All, sync::StallCondition::All>();
    sync::wait::stall<sync::StallTarget::HardwareDefault, sync::StallCondition::HardwareDefault>();
    sync::wait::stall<sync::StallTarget::Thcon, sync::StallCondition::SrcAValid | sync::StallCondition::SrcBValid>();
}

extern "C" __attribute__((noinline, used)) void reference_stall_variants()
{
    TTI_STALLWAIT(384 /*stall_res*/, 2064 /*wait_res*/);
    TTI_STALLWAIT(511 /*stall_res*/, 8191 /*wait_res*/);
    TTI_STALLWAIT(0 /*stall_res*/, 0 /*wait_res*/);
    TTI_STALLWAIT(32 /*stall_res*/, 384 /*wait_res*/);
}

extern "C" __attribute__((noinline, used)) void semaphore_wait_variants()
{
    sync::wait::semaphore<sync::StallTarget::Sync | sync::StallTarget::Math, sync::SemaphoreCondition::WhileZero, sync::Semaphore::S1>();
    sync::wait::semaphore<sync::StallTarget::Pack, sync::SemaphoreCondition::WhileMaximum, sync::Semaphore::S0, sync::Semaphore::S7>();
    sync::wait::semaphore<
        sync::StallTarget::Unpack,
        sync::SemaphoreCondition::WhileZero | sync::SemaphoreCondition::WhileMaximum,
        sync::Semaphore::S0,
        sync::Semaphore::S1,
        sync::Semaphore::S2,
        sync::Semaphore::S3,
        sync::Semaphore::S4,
        sync::Semaphore::S5,
        sync::Semaphore::S6,
        sync::Semaphore::S7>();
}

extern "C" __attribute__((noinline, used)) void reference_semaphore_wait_variants()
{
    TTI_SEMWAIT(66 /*stall_res*/, 0x02 /*sem_sel*/, 1 /*wait_sem_cond*/);
    TTI_SEMWAIT(4 /*stall_res*/, 0x81 /*sem_sel*/, 2 /*wait_sem_cond*/);
    TTI_SEMWAIT(8 /*stall_res*/, 0xff /*sem_sel*/, 3 /*wait_sem_cond*/);
}

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN((sync::wait::stall_operation<sync::StallTarget::Pack, sync::StallCondition::PackerIdle>()));
    TTI_INSN((sync::wait::semaphore_operation<sync::StallTarget::Mover, sync::SemaphoreCondition::WhileZero, sync::Semaphore::S2>()));
}

extern "C" __attribute__((noinline, used)) void reference_issue_encoded_operations()
{
    TTI_STALLWAIT(4 /*stall_res*/, 8 /*wait_res*/);
    TTI_SEMWAIT(16 /*stall_res*/, 0x04 /*sem_sel*/, 1 /*wait_sem_cond*/);
}

extern "C" __attribute__((noinline, used)) void stall_runtime(const sync::StallTarget targets, const sync::StallCondition conditions)
{
    sync::wait::stall(targets, conditions);
}

extern "C" __attribute__((noinline, used)) void reference_stall_runtime(const sync::StallTarget targets, const sync::StallCondition conditions)
{
    TT_STALLWAIT(hal::to_underlying(targets), hal::to_underlying(conditions));
}

extern "C" __attribute__((noinline, used)) void semaphore_runtime(
    const sync::StallTarget targets, const sync::Semaphore semaphore, const sync::SemaphoreCondition conditions)
{
    sync::wait::semaphore(targets, semaphore, conditions);
}

extern "C" __attribute__((noinline, used)) void reference_semaphore_runtime(
    const sync::StallTarget targets, const sync::Semaphore semaphore, const sync::SemaphoreCondition conditions)
{
    TT_SEMWAIT(hal::to_underlying(targets), 1u << hal::to_underlying(semaphore), hal::to_underlying(conditions));
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime operands must trap before the immediate instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_stall_targets()
{
    sync::wait::stall(static_cast<sync::StallTarget>(512), sync::StallCondition::MathIdle);
}

// ASSERT-LABEL: <reject_invalid_runtime_stall_targets>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_stall_conditions()
{
    sync::wait::stall(sync::StallTarget::Math, static_cast<sync::StallCondition>(8192));
}

// ASSERT-LABEL: <reject_invalid_runtime_stall_conditions>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_stall_boundaries()
{
    sync::wait::stall(sync::StallTarget::All, sync::StallCondition::All);
}

// ASSERT-LABEL: <accept_runtime_stall_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
