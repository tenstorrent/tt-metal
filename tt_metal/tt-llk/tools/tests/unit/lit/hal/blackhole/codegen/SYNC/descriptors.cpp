// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %t/codegen.cpp -o %t/unpack.o
// RUN: %{blackhole_objdump} -drz %t/unpack.o > %t/unpack.dump
// RUN: %{blackhole_compare_codegen} %t/unpack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fno-ipa-icf -c %t/codegen.cpp -o %t/math.o
// RUN: %{blackhole_objdump} -drz %t/math.o > %t/math.dump
// RUN: %{blackhole_compare_codegen} %t/math.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -fno-ipa-icf -c %t/codegen.cpp -o %t/pack.o
// RUN: %{blackhole_objdump} -drz %t/pack.o > %t/pack.dump
// RUN: %{blackhole_compare_codegen} %t/pack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

//--- common.h
#pragma once

#include <cstdint>

#include "hal/sync.h"

namespace hs = hal::sync;

//--- codegen.cpp
#include "common.h"

// Encoding must match raw instruction words without issuing instructions or MMIO.

extern "C" __attribute__((noinline, used)) std::uint32_t encode_mutex_acquire(hs::Mutex selector)
{
    return hs::MutexAcquire {selector}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_mutex_acquire(hs::Mutex selector)
{
    return TT_OP_ATGETM(hal::to_underlying(selector));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_mutex_release(hs::Mutex selector)
{
    return hs::MutexRelease {selector}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_mutex_release(hs::Mutex selector)
{
    return TT_OP_ATRELM(hal::to_underlying(selector));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_init(hs::SemaphoreInit descriptor)
{
    return descriptor.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_init(hs::SemaphoreInit descriptor)
{
    return TT_OP_SEMINIT(descriptor.maximum, descriptor.initial, descriptor.semaphores.mask());
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_post(hs::Semaphore selector)
{
    return hs::SemaphorePost {selector}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_post(hs::Semaphore selector)
{
    return TT_OP_SEMPOST((1u << hal::to_underlying(selector)));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_get(hs::Semaphore selector)
{
    return hs::SemaphoreGet {selector}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_get(hs::Semaphore selector)
{
    return TT_OP_SEMGET((1u << hal::to_underlying(selector)));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_stall_wait(hs::StallTarget targets, hs::StallCondition conditions)
{
    return hs::StallWait {targets, conditions}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_stall_wait(hs::StallTarget targets, hs::StallCondition conditions)
{
    return TT_OP_STALLWAIT(hal::to_underlying(targets), hal::to_underlying(conditions));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_wait(
    hs::StallTarget targets, hs::Semaphore selector, hs::SemaphoreCondition conditions)
{
    return hs::SemaphoreWait {targets, selector, conditions}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_wait(
    hs::StallTarget targets, hs::Semaphore selector, hs::SemaphoreCondition conditions)
{
    return TT_OP_SEMWAIT(hal::to_underlying(targets), (1u << hal::to_underlying(selector)), hal::to_underlying(conditions));
}

// Encoded words also work as immediate Tensix instructions.
extern "C" __attribute__((noinline, used)) void issue_sync_descriptor()
{
    constexpr auto word = hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S1, hs::SemaphoreCondition::WhileZero}.operation();
    TTI_INSN(word);
}

extern "C" __attribute__((noinline, used)) void reference_issue_sync_descriptor()
{
    TTI_SEMWAIT(64 /*stall_res*/, 0x02 /*sem_sel*/, 1 /*wait_sem_cond*/);
}

// Tensix forms combine selector lists into one mask.
extern "C" __attribute__((noinline, used)) void issue_semaphore_selectors()
{
    hs::semaphore::get<hs::Semaphore::S1, hs::Semaphore::S3>();
    hs::semaphore::post<hs::Semaphore::S0, hs::Semaphore::S7>();
}

extern "C" __attribute__((noinline, used)) void reference_issue_semaphore_selectors()
{
    TTI_SEMGET(0x0a /*sem_sel*/);
    TTI_SEMPOST(0x81 /*sem_sel*/);
}

extern "C" __attribute__((noinline, used)) void issue_runtime_semaphore_selectors(hs::Semaphore first, hs::Semaphore second)
{
    hs::semaphore::post({first, second});
    hs::semaphore::init({first, second}, 5, 15);
    hs::wait::semaphore(hs::StallTarget::Math, {first, second}, hs::SemaphoreCondition::WhileZero);
    hs::semaphore::get<hs::Access::Tensix>(second);
    hs::semaphore::post<hs::Access::Tensix>(second);
}

extern "C" __attribute__((noinline, used)) void reference_issue_runtime_semaphore_selectors(hs::Semaphore first, hs::Semaphore second)
{
    const auto first_bit  = 1u << hal::to_underlying(first);
    const auto second_bit = 1u << hal::to_underlying(second);
    TT_SEMPOST(first_bit | second_bit);
    TT_SEMINIT(15 /*max_value*/, 5 /*init_value*/, first_bit | second_bit);
    TT_SEMWAIT(64 /*stall_res*/, first_bit | second_bit, 1 /*wait_sem_cond*/);
    TT_SEMGET(second_bit);
    TT_SEMPOST(second_bit);
}

// Explicit MMIO access uses the same selector as an index, never as a mask.
extern "C" __attribute__((noinline, used)) void issue_mmio_semaphore_selectors(hs::Semaphore selector)
{
    hs::semaphore::get<hs::Access::MMIO, hs::Semaphore::S1>();
    hs::semaphore::post<hs::Access::MMIO, hs::Semaphore::S7>();
    hs::semaphore::get<hs::Access::MMIO>(selector);
    hs::semaphore::post<hs::Access::MMIO>(selector);
}

// TODO: #58443 Replace the ckernel:: MMIO references with raw pc_buf_base accesses.
extern "C" __attribute__((noinline, used)) void reference_issue_mmio_semaphore_selectors(hs::Semaphore selector)
{
    ckernel::semaphore_get(1 /*index*/);
    ckernel::semaphore_post(7 /*index*/);
    ckernel::semaphore_get(hal::to_underlying(selector));
    ckernel::semaphore_post(hal::to_underlying(selector));
}

extern "C" __attribute__((noinline, used)) std::uint8_t read_mmio_semaphore()
{
    return hs::semaphore::read<hs::Semaphore::S7>();
}

extern "C" __attribute__((noinline, used)) std::uint8_t reference_read_mmio_semaphore()
{
    return ckernel::semaphore_read(7 /*index*/);
}

extern "C" __attribute__((noinline, used)) std::uint8_t read_runtime_mmio_semaphore(hs::Semaphore selector)
{
    return hs::semaphore::read(selector);
}

extern "C" __attribute__((noinline, used)) std::uint8_t reference_read_runtime_mmio_semaphore(hs::Semaphore selector)
{
    return ckernel::semaphore_read(hal::to_underlying(selector));
}

//--- asserts.cpp
#include "common.h"

// Each invalid runtime operand must emit its own ebreak. One probe per check
// keeps the expectation independent of how many checks an operand violates.

extern "C" void reject_invalid_mutex_acquire()
{
    (void)hs::MutexAcquire {static_cast<hs::Mutex>(1)}.operation();
}

// ASSERT-LABEL: <reject_invalid_mutex_acquire>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_mutex_release()
{
    (void)hs::MutexRelease {static_cast<hs::Mutex>(5)}.operation();
}

// ASSERT-LABEL: <reject_invalid_mutex_release>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_semaphore_selector()
{
    (void)hs::SemaphorePost {static_cast<hs::Semaphore>(8)}.operation();
}

// ASSERT-LABEL: <reject_invalid_semaphore_selector>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_semaphore_initial()
{
    (void)hs::SemaphoreInit {hs::Semaphore::S0, 16, 15}.operation();
}

// ASSERT-LABEL: <reject_invalid_semaphore_initial>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_semaphore_maximum()
{
    (void)hs::SemaphoreInit {hs::Semaphore::S0, 0, 16}.operation();
}

// ASSERT-LABEL: <reject_invalid_semaphore_maximum>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_semaphore_initial_above_maximum()
{
    (void)hs::SemaphoreInit {hs::Semaphore::S0, 5, 4}.operation();
}

// ASSERT-LABEL: <reject_semaphore_initial_above_maximum>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_stall_targets()
{
    (void)hs::StallWait {static_cast<hs::StallTarget>(512), hs::StallCondition::MathIdle}.operation();
}

// ASSERT-LABEL: <reject_invalid_stall_targets>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_stall_conditions()
{
    (void)hs::StallWait {hs::StallTarget::Math, static_cast<hs::StallCondition>(8192)}.operation();
}

// ASSERT-LABEL: <reject_invalid_stall_conditions>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_invalid_semaphore_wait_targets()
{
    (void)hs::SemaphoreWait {static_cast<hs::StallTarget>(512), hs::Semaphore::S0, hs::SemaphoreCondition::WhileZero}.operation();
}

// ASSERT-LABEL: <reject_invalid_semaphore_wait_targets>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_empty_semaphore_wait_conditions()
{
    (void)hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S0, static_cast<hs::SemaphoreCondition>(0)}.operation();
}

// ASSERT-LABEL: <reject_empty_semaphore_wait_conditions>:
// ASSERT: ebreak
// ASSERT: ret

extern "C" void reject_out_of_range_semaphore_wait_conditions()
{
    (void)hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S0, static_cast<hs::SemaphoreCondition>(4)}.operation();
}

// ASSERT-LABEL: <reject_out_of_range_semaphore_wait_conditions>:
// ASSERT: ebreak
// ASSERT: ret
