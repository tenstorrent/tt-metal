// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// LLK_ASSERT is disabled so runtime paths contain only the issued instruction.

#include "hal/sync.h"

namespace sync = hal::sync;

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
