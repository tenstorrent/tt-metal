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
    return TT_OP_SEMINIT(descriptor.maximum, descriptor.initial, hal::to_underlying(descriptor.mask));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_post(hs::SemaphoreMask mask)
{
    return hs::SemaphorePost {mask}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_post(hs::SemaphoreMask mask)
{
    return TT_OP_SEMPOST(hal::to_underlying(mask));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_semaphore_get(hs::SemaphoreMask mask)
{
    return hs::SemaphoreGet {mask}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_get(hs::SemaphoreMask mask)
{
    return TT_OP_SEMGET(hal::to_underlying(mask));
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
    hs::StallTarget targets, hs::SemaphoreMask mask, hs::SemaphoreCondition conditions)
{
    return hs::SemaphoreWait {targets, mask, conditions}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_semaphore_wait(
    hs::StallTarget targets, hs::SemaphoreMask mask, hs::SemaphoreCondition conditions)
{
    return TT_OP_SEMWAIT(hal::to_underlying(targets), hal::to_underlying(mask), hal::to_underlying(conditions));
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_stream_wait(hs::StreamWait descriptor)
{
    return descriptor.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_stream_wait(hs::StreamWait descriptor)
{
    return TT_OP_STREAMWAIT(
        hal::to_underlying(descriptor.targets), descriptor.target_low, hal::to_underlying(descriptor.target), hal::to_underlying(descriptor.slot));
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
    constexpr auto word = hs::SemaphoreWait {hs::StallTarget::Math, hs::SemaphoreMask::S1, hs::SemaphoreCondition::WhileZero}.operation();
    TTI_INSN(word);
}

extern "C" __attribute__((noinline, used)) void reference_issue_sync_descriptor()
{
    TTI_SEMWAIT(64, 2, 1);
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
    (void)hs::SemaphorePost {hs::SemaphoreMask::None}.operation();
    (void)hs::SemaphoreInit {hs::SemaphoreMask::S0, 16, 15}.operation();
    (void)hs::SemaphoreInit {hs::SemaphoreMask::S0, 0, 16}.operation();
    (void)hs::StallWait {static_cast<hs::StallTarget>(512), hs::StallCondition::MathIdle}.operation();
    (void)hs::StallWait {hs::StallTarget::Math, static_cast<hs::StallCondition>(8192)}.operation();
    (void)hs::SemaphoreWait {hs::StallTarget::Math, hs::SemaphoreMask::S0, static_cast<hs::SemaphoreCondition>(0)}.operation();
    (void)hs::StreamWait {hs::StallTarget::Unpack, static_cast<hs::StreamSlot>(4), hs::StreamTarget::Phase, 0}.operation();
    (void)hs::StreamWait {hs::StallTarget::Unpack, hs::StreamSlot::S0, static_cast<hs::StreamTarget>(2), 0}.operation();
    (void)hs::StreamWait {hs::StallTarget::Unpack, hs::StreamSlot::S0, hs::StreamTarget::Phase, 1024}.operation();
    (void)hm::FlushTdma {static_cast<hm::FlushScope>(16)}.operation();
    (void)hm::ResourceDeclaration {16, 0, 1}.operation();
    (void)hm::ResourceDeclaration {0, 512, 1}.operation();
    (void)hm::ResourceDeclaration {0, 0, 2048}.operation();
}

// ASSERT-LABEL: <reject_invalid_runtime_descriptors>:
// ASSERT-COUNT-14: ebreak
// ASSERT-NOT: ebreak
// ASSERT-NOT: sw
// ASSERT: ret

#endif
