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

//--- common.h
#pragma once

#include "hal/sync.h"

namespace sync = hal::sync;

//--- codegen.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) void acquire_release_physical()
{
    sync::mutex::acquire<sync::Mutex::M0>();
    sync::mutex::acquire<sync::Mutex::M2>();
    sync::mutex::release<sync::Mutex::M3>();
    sync::mutex::release<sync::Mutex::M4>();
}

extern "C" __attribute__((noinline, used)) void reference_acquire_release_physical()
{
    TTI_ATGETM(0 /*mutex_index*/);
    TTI_ATGETM(2 /*mutex_index*/);
    TTI_ATRELM(3 /*mutex_index*/);
    TTI_ATRELM(4 /*mutex_index*/);
}

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN(sync::mutex::acquire_operation<sync::Mutex::M4>());
    TTI_INSN(sync::mutex::release_operation<sync::Mutex::M0>());
}

extern "C" __attribute__((noinline, used)) void reference_issue_encoded_operations()
{
    TTI_ATGETM(4 /*mutex_index*/);
    TTI_ATRELM(0 /*mutex_index*/);
}

// Runtime forms push the mutex index through the instruction buffer.

extern "C" __attribute__((noinline, used)) void acquire_runtime(const sync::Mutex mutex)
{
    sync::mutex::acquire(mutex);
}

extern "C" __attribute__((noinline, used)) void reference_acquire_runtime(const sync::Mutex mutex)
{
    TT_ATGETM(hal::to_underlying(mutex));
}

extern "C" __attribute__((noinline, used)) void release_runtime(const sync::Mutex mutex)
{
    sync::mutex::release(mutex);
}

extern "C" __attribute__((noinline, used)) void reference_release_runtime(const sync::Mutex mutex)
{
    TT_ATRELM(hal::to_underlying(mutex));
}

//--- asserts.cpp
#include "common.h"

// Invalid runtime operands must trap before the immediate instruction is issued;
// valid boundaries must not trap.

extern "C" void reject_invalid_runtime_acquire()
{
    sync::mutex::acquire(static_cast<sync::Mutex>(1));
}

// ASSERT-LABEL: <reject_invalid_runtime_acquire>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_runtime_release()
{
    sync::mutex::release(static_cast<sync::Mutex>(5));
}

// ASSERT-LABEL: <reject_invalid_runtime_release>:
// ASSERT-NOT: {{[[:space:]]tt}}
// ASSERT: ebreak
// ASSERT: ret

extern "C" void accept_runtime_mutex_boundaries()
{
    sync::mutex::acquire(sync::Mutex::M0);
    sync::mutex::acquire(sync::Mutex::M4);
    sync::mutex::release(sync::Mutex::M0);
    sync::mutex::release(sync::Mutex::M4);
}

// ASSERT-LABEL: <accept_runtime_mutex_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
