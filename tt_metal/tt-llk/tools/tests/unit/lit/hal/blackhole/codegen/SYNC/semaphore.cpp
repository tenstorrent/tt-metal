// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// LLK_ASSERT is disabled so runtime paths contain only the issued instruction.
// Semaphore selectors are one bit per semaphore: S<n> is bit n.

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

extern "C" __attribute__((noinline, used)) void init_post_get_selectors()
{
    sync::semaphore::init<1, 2, sync::Semaphore::S1, sync::Semaphore::S4>();
    sync::semaphore::init<
        0,
        15,
        sync::Semaphore::S0,
        sync::Semaphore::S1,
        sync::Semaphore::S2,
        sync::Semaphore::S3,
        sync::Semaphore::S4,
        sync::Semaphore::S5,
        sync::Semaphore::S6,
        sync::Semaphore::S7>();
    sync::semaphore::post<sync::Semaphore::S0>();
    sync::semaphore::get<sync::Semaphore::S7>();
}

extern "C" __attribute__((noinline, used)) void reference_init_post_get_selectors()
{
    TTI_SEMINIT(2 /*max_value*/, 1 /*init_value*/, 0x12 /*sem_sel*/);
    TTI_SEMINIT(15 /*max_value*/, 0 /*init_value*/, 0xff /*sem_sel*/);
    TTI_SEMPOST(0x01 /*sem_sel*/);
    TTI_SEMGET(0x80 /*sem_sel*/);
}

extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN((sync::semaphore::init_operation<3, 4, sync::Semaphore::S5>()));
    TTI_INSN((sync::semaphore::post_operation<sync::Semaphore::S6, sync::Semaphore::S7>()));
    TTI_INSN(sync::semaphore::get_operation<sync::Semaphore::S2>());
}

extern "C" __attribute__((noinline, used)) void reference_issue_encoded_operations()
{
    TTI_SEMINIT(4 /*max_value*/, 3 /*init_value*/, 0x20 /*sem_sel*/);
    TTI_SEMPOST(0xc0 /*sem_sel*/);
    TTI_SEMGET(0x04 /*sem_sel*/);
}

extern "C" __attribute__((noinline, used)) void post_get_tensix_index()
{
    sync::semaphore::post<sync::Access::Tensix, sync::Semaphore::S1>();
    sync::semaphore::get<sync::Access::Tensix, sync::Semaphore::S7>();
}

extern "C" __attribute__((noinline, used)) void reference_post_get_tensix_index()
{
    TTI_SEMPOST(0x02 /*sem_sel*/);
    TTI_SEMGET(0x80 /*sem_sel*/);
}

// Runtime forms push the encoded word through the instruction buffer. The SEMINIT
// references add the maximum value (bits 23:20) separately, which avoids an
// incidental reassociation difference from the HAL body.

extern "C" __attribute__((noinline, used)) void init_runtime(const sync::Semaphore semaphore, const std::uint32_t initial, const std::uint32_t maximum)
{
    sync::semaphore::init(semaphore, initial, maximum);
}

extern "C" __attribute__((noinline, used)) void reference_init_runtime(
    const sync::Semaphore semaphore, const std::uint32_t initial, const std::uint32_t maximum)
{
    const std::uint32_t selector = 1u << hal::to_underlying(semaphore);
    TT_INSN(TT_OP_SEMINIT(0 /*max_value*/, initial, selector) + (maximum << 20));
}

extern "C" __attribute__((noinline, used)) void post_runtime(const sync::Semaphore semaphore)
{
    sync::semaphore::post(semaphore);
}

extern "C" __attribute__((noinline, used)) void reference_post_runtime(const sync::Semaphore semaphore)
{
    TT_SEMPOST(1u << hal::to_underlying(semaphore));
}

extern "C" __attribute__((noinline, used)) void get_runtime(const sync::Semaphore semaphore)
{
    sync::semaphore::get(semaphore);
}

extern "C" __attribute__((noinline, used)) void reference_get_runtime(const sync::Semaphore semaphore)
{
    TT_SEMGET(1u << hal::to_underlying(semaphore));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_init_runtime(
    const sync::Semaphore semaphore, const std::uint32_t initial, const std::uint32_t maximum)
{
    return sync::semaphore::init_operation(semaphore, initial, maximum);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_init_runtime(
    const sync::Semaphore semaphore, const std::uint32_t initial, const std::uint32_t maximum)
{
    const std::uint32_t selector = 1u << hal::to_underlying(semaphore);
    return TT_OP_SEMINIT(0 /*max_value*/, initial, selector) + (maximum << 20);
}
