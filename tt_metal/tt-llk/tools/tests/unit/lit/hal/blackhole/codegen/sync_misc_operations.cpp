// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -fno-ipa-icf -c %s -o %t.unpack.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.unpack.o > %t.unpack.dump
// RUN: %{blackhole_compare_codegen} %t.unpack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fno-ipa-icf -c %s -o %t.math.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.math.o > %t.math.dump
// RUN: %{blackhole_compare_codegen} %t.math.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -fno-ipa-icf -c %s -o %t.pack.o
// RUN: %{blackhole_objdump} -t --special-syms -drz %t.pack.o > %t.pack.dump
// RUN: %{blackhole_compare_codegen} %t.pack.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT -c %s -o %t.assert.o
// RUN: %{blackhole_objdump} -d %t.assert.o | FileCheck %s --check-prefix=ASSERT

#include <cstdint>

#include "hal/misc.h"
#include "hal/sync.h"

namespace hs = hal::sync;
namespace hm = hal::misc;

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

extern "C" __attribute__((noinline, used)) std::uint32_t encode_flush_tdma(hm::FlushScope scope)
{
    return hm::FlushTdma {scope}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_flush_tdma(hm::FlushScope scope)
{
    return TT_OP_FLUSHDMA(hal::to_underlying(scope));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_reset_tdma()
{
    return hm::ResetTdma {}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_reset_tdma()
{
    return TT_OP_RSTDMA;
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_tbuf_command()
{
    return hm::TbufCommand {}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_tbuf_command()
{
    return TT_OP_TBUFCMD;
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_resource_declaration(hm::ResourceDeclaration descriptor)
{
    return descriptor.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_resource_declaration(hm::ResourceDeclaration descriptor)
{
    return TT_OP_RESOURCEDECL(descriptor.linger_time, descriptor.resources, descriptor.instruction_class);
}

// Encoded words also work as immediate Tensix instructions.
extern "C" __attribute__((noinline, used)) void issue_sync_descriptor()
{
    constexpr auto word = hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S1, hs::SemaphoreCondition::WhileZero}.operation();
    TTI_INSN(word);
}

extern "C" __attribute__((noinline, used)) void reference_issue_sync_descriptor()
{
    TTI_SEMWAIT(64, 2, 1);
}

// The default path derives a Tensix mask from one or more semaphore indices.
extern "C" __attribute__((noinline, used)) void issue_semaphore_selectors()
{
    hs::semaphore::get<hs::Semaphore::S1>();
    hs::semaphore::get<hs::Semaphore::S1, hs::Semaphore::S3>();
    hs::semaphore::post<hs::Semaphore::S0, hs::Semaphore::S7>();
    hs::semaphore::init<5, 15, hs::Semaphore::S0, hs::Semaphore::S7>();
    hs::wait::semaphore<hs::StallTarget::Math, hs::SemaphoreCondition::WhileZero, hs::Semaphore::S1, hs::Semaphore::S3>();
    hs::semaphore::get<hs::Access::Tensix, hs::Semaphore::S1>();
    hs::semaphore::post<hs::Access::Tensix, hs::Semaphore::S7>();
}

extern "C" __attribute__((noinline, used)) void reference_issue_semaphore_selectors()
{
    TTI_SEMGET(2);
    TTI_SEMGET(10);
    TTI_SEMPOST(129);
    TTI_SEMINIT(15, 5, 129);
    TTI_SEMWAIT(64, 10, 1);
    TTI_SEMGET(2);
    TTI_SEMPOST(128);
}

extern "C" __attribute__((noinline, used)) void issue_runtime_semaphore_selectors(hs::Semaphore first, hs::Semaphore second)
{
    hs::semaphore::get(first);
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
    TT_SEMGET(first_bit);
    TT_SEMPOST(first_bit | second_bit);
    TT_SEMINIT(15, 5, first_bit | second_bit);
    TT_SEMWAIT(64, first_bit | second_bit, 1);
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

extern "C" __attribute__((noinline, used)) void reference_issue_mmio_semaphore_selectors(hs::Semaphore selector)
{
    ckernel::semaphore_get(1);
    ckernel::semaphore_post(7);
    ckernel::semaphore_get(hal::to_underlying(selector));
    ckernel::semaphore_post(hal::to_underlying(selector));
}

extern "C" __attribute__((noinline, used)) void issue_misc_descriptors()
{
    hm::run<hm::FlushTdma {hm::FlushScope::Packer}>();
    hm::run<hm::ResetTdma {}>();
    hm::run<hm::TbufCommand {}>();
    hm::run<hm::ResourceDeclaration {3, 5, 7}>();
}

extern "C" __attribute__((noinline, used)) void reference_issue_misc_descriptors()
{
    TTI_FLUSHDMA(8);
    TTI_RSTDMA;
    TTI_TBUFCMD;
    TTI_RESOURCEDECL(7, 5, 3);
}

extern "C" __attribute__((noinline, used)) void issue_runtime_misc_descriptor(hm::ResourceDeclaration descriptor)
{
    hm::run(descriptor);
}

extern "C" __attribute__((noinline, used)) void reference_issue_runtime_misc_descriptor(hm::ResourceDeclaration descriptor)
{
    TT_RESOURCEDECL(descriptor.linger_time, descriptor.resources, descriptor.instruction_class);
}

#ifdef ENABLE_LLK_ASSERT

extern "C" void reject_invalid_runtime_descriptors()
{
    (void)hs::MutexAcquire {static_cast<hs::Mutex>(1)}.operation();
    (void)hs::SemaphorePost {static_cast<hs::Semaphore>(8)}.operation();
    (void)hs::SemaphoreInit {hs::Semaphore::S0, 16, 15}.operation();
    (void)hs::SemaphoreInit {hs::Semaphore::S0, 0, 16}.operation();
    (void)hs::StallWait {static_cast<hs::StallTarget>(512), hs::StallCondition::MathIdle}.operation();
    (void)hs::StallWait {hs::StallTarget::Math, static_cast<hs::StallCondition>(8192)}.operation();
    (void)hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S0, static_cast<hs::SemaphoreCondition>(0)}.operation();
    (void)hm::FlushTdma {static_cast<hm::FlushScope>(16)}.operation();
    (void)hm::ResourceDeclaration {16, 0, 1}.operation();
    (void)hm::ResourceDeclaration {0, 512, 1}.operation();
    (void)hm::ResourceDeclaration {0, 0, 2048}.operation();
}

// ASSERT-LABEL: <reject_invalid_runtime_descriptors>:
// ASSERT-COUNT-11: ebreak
// ASSERT-NOT: ebreak
// ASSERT-NOT: sw
// ASSERT: ret

#endif
