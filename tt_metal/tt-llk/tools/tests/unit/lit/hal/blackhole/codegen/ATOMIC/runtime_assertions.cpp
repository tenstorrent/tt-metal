// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// Check assertion branches separately from the TT_* encoding comparisons.
// These are compile-only checks; ebreak is inspected, not executed.

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_fetch_increment_descriptor(const std::uint32_t address)
{
    const auto descriptor = atomic::FetchIncrement {hal::gpr(address), hal::gpr<2>(), std::uint8_t {1}, std::uint8_t {8}};
    return descriptor.operation();
}

// CHECK-LABEL: <runtime_fetch_increment_descriptor>:
// CHECK: li [[LIMIT:a[0-7]]],63
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void counter_runtime_word(const std::uint8_t word)
{
    const auto counter = atomic::counter<16>(hal::gpr<1>(), word);
    counter.fetch_add(hal::gpr<2>());
}

// CHECK-LABEL: <counter_runtime_word>:
// CHECK: li [[LIMIT:a[0-7]]],3
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_increment(const std::uint8_t increment_log2)
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    return fifo.operation().pop(hal::gpr<4>(), increment_log2);
}

// CHECK-LABEL: <fifo_runtime_increment>:
// CHECK: li [[LIMIT:a[0-7]]],8
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_push_default(const std::uint32_t address)
{
    const auto fifo = atomic::fifo<8>(hal::gpr(address));
    fifo.push(hal::gpr<4>());
}

// CHECK-LABEL: <fifo_runtime_push_default>:
// CHECK: li [[LIMIT:a[0-7]]],63
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_wait_full(const std::uint32_t result)
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr(result));
}

// CHECK-LABEL: <fifo_runtime_wait_full>:
// CHECK: li [[LIMIT:a[0-7]]],63
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_acquire_release(const std::uint32_t address)
{
    const auto lock = atomic::Lock {hal::gpr(address), 1};
    lock.acquire();
    lock.release();
}

// CHECK-LABEL: <lock_runtime_acquire_release>:
// CHECK: li [[LIMIT:a[0-7]]],63
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_compare_set(const std::uint8_t compare, const std::uint8_t set)
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(compare, set);
}

// CHECK-LABEL: <lock_runtime_compare_set>:
// CHECK: li [[LIMIT:a[0-7]]],15
// CHECK: bltu [[LIMIT]],a0,
// CHECK: li [[LIMIT]],15
// CHECK: bltu [[LIMIT]],a1,
// CHECK-COUNT-2: ebreak

extern "C" __attribute__((noinline, used)) void masked_store_runtime_mask(const std::uint8_t mask)
{
    atomic::store_masked(hal::gpr<8>(), hal::gpr<12>(), mask);
}

// CHECK-LABEL: <masked_store_runtime_mask>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_capacity(const std::uint8_t capacity_log2)
{
    const auto descriptor = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, capacity_log2, std::uint8_t {0}, false};
    return descriptor.operation();
}

// CHECK-LABEL: <fifo_runtime_capacity>:
// CHECK: li [[LIMIT:a[0-7]]],14
// CHECK: bltu [[LIMIT]],a0,
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void fifo_capacity15_updates()
{
    constexpr auto fifo = atomic::fifo<15>(hal::gpr<3>());
    fifo.pop(hal::gpr<4>());
    fifo.push(hal::gpr<4>());
}

// CHECK-LABEL: <fifo_capacity15_updates>:
// CHECK: ebreak
// CHECK: ttatincgetptr
// CHECK: ebreak
// CHECK: ttatincgetptr
// CHECK: ret

extern "C" __attribute__((noinline, used)) void fifo_capacity15_waits()
{
    constexpr auto fifo = atomic::fifo<15>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Empty>(hal::gpr<4>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr<4>());
}

// CHECK-LABEL: <fifo_capacity15_waits>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void invalid_static_counter()
{
    constexpr auto counter = atomic::counter<0>(hal::gpr<1>());
    counter.fetch_add(hal::gpr<2>());
}

// CHECK-LABEL: <invalid_static_counter>:
// CHECK: ebreak
// CHECK: ret

extern "C" __attribute__((noinline, used)) void invalid_static_address()
{
    constexpr auto counter = atomic::counter<32>(hal::gpr<64>());
    counter.fetch_add(hal::gpr<2>());
}

// CHECK-LABEL: <invalid_static_address>:
// CHECK: ebreak
// CHECK: ret

extern "C" __attribute__((noinline, used)) void invalid_static_quad()
{
    atomic::store_masked(hal::gpr<3>(), hal::gpr<5>(), 0xff);
}

// CHECK-LABEL: <invalid_static_quad>:
// CHECK: ebreak
// CHECK: ret
