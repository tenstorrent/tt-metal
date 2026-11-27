// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

// Each runtime check is called with the last valid and the first invalid value, which the compiler folds.
// The test checks only whether the LLK_ASSERT ebreak is emitted, so it does not depend on how the comparison is compiled.

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_fetch_increment_descriptor_valid()
{
    const auto descriptor = atomic::FetchIncrement {hal::gpr(std::uint32_t {63}), hal::gpr<2>(), std::uint8_t {1}, std::uint8_t {8}};
    return descriptor.operation();
}

// CHECK-LABEL: <runtime_fetch_increment_descriptor_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_fetch_increment_descriptor_invalid()
{
    const auto descriptor = atomic::FetchIncrement {hal::gpr(std::uint32_t {64}), hal::gpr<2>(), std::uint8_t {1}, std::uint8_t {8}};
    return descriptor.operation();
}

// CHECK-LABEL: <runtime_fetch_increment_descriptor_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void counter_runtime_word_valid()
{
    const auto counter = atomic::counter<16>(hal::gpr<1>(), std::uint8_t {3});
    counter.fetch_add(hal::gpr<2>());
}

// CHECK-LABEL: <counter_runtime_word_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void counter_runtime_word_invalid()
{
    const auto counter = atomic::counter<16>(hal::gpr<1>(), std::uint8_t {4});
    counter.fetch_add(hal::gpr<2>());
}

// CHECK-LABEL: <counter_runtime_word_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_increment_valid()
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    return fifo.operation().pop(hal::gpr<4>(), std::uint8_t {8});
}

// CHECK-LABEL: <fifo_runtime_increment_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_increment_invalid()
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    return fifo.operation().pop(hal::gpr<4>(), std::uint8_t {9});
}

// CHECK-LABEL: <fifo_runtime_increment_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_push_default_valid()
{
    const auto fifo = atomic::fifo<8>(hal::gpr(std::uint32_t {63}));
    fifo.push(hal::gpr<4>());
}

// CHECK-LABEL: <fifo_runtime_push_default_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_push_default_invalid()
{
    const auto fifo = atomic::fifo<8>(hal::gpr(std::uint32_t {64}));
    fifo.push(hal::gpr<4>());
}

// CHECK-LABEL: <fifo_runtime_push_default_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_wait_full_valid()
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr(std::uint32_t {63}));
}

// CHECK-LABEL: <fifo_runtime_wait_full_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_wait_full_invalid()
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr(std::uint32_t {64}));
}

// CHECK-LABEL: <fifo_runtime_wait_full_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_acquire_release_valid()
{
    const auto lock = atomic::Lock {hal::gpr(std::uint32_t {63}), 1 /*word*/};
    lock.acquire();
    lock.release();
}

// CHECK-LABEL: <lock_runtime_acquire_release_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_acquire_release_invalid()
{
    const auto lock = atomic::Lock {hal::gpr(std::uint32_t {64}), 1 /*word*/};
    lock.acquire();
    lock.release();
}

// CHECK-LABEL: <lock_runtime_acquire_release_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_compare_valid()
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(std::uint8_t {15}, std::uint8_t {15});
}

// CHECK-LABEL: <lock_runtime_compare_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_compare_invalid()
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(std::uint8_t {16}, std::uint8_t {15});
}

// CHECK-LABEL: <lock_runtime_compare_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_set_valid()
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(std::uint8_t {15}, std::uint8_t {15});
}

// CHECK-LABEL: <lock_runtime_set_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void lock_runtime_set_invalid()
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(std::uint8_t {15}, std::uint8_t {16});
}

// CHECK-LABEL: <lock_runtime_set_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void masked_store_runtime_mask(const std::uint8_t mask)
{
    atomic::store_masked(hal::gpr<8>(), hal::gpr<12>(), mask);
}

// CHECK-LABEL: <masked_store_runtime_mask>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_capacity_valid()
{
    const auto descriptor = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, std::uint8_t {14}, std::uint8_t {0}, false};
    return descriptor.operation();
}

// CHECK-LABEL: <fifo_runtime_capacity_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_capacity_invalid()
{
    const auto descriptor = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, std::uint8_t {15}, std::uint8_t {0}, false};
    return descriptor.operation();
}

// CHECK-LABEL: <fifo_runtime_capacity_invalid>:
// CHECK: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_set_address_valid()
{
    constexpr auto fifo = atomic::fifo<3>(hal::gpr<3>());
    fifo.set_address(std::uint32_t {0x1010});
}

// CHECK-LABEL: <fifo_runtime_set_address_valid>:
// CHECK-NOT: ebreak
// CHECK: ret
// CHECK-NOT: ebreak

extern "C" __attribute__((noinline, used)) void fifo_runtime_set_address_invalid()
{
    constexpr auto fifo = atomic::fifo<3>(hal::gpr<3>());
    fifo.set_address(std::uint32_t {0x1018});
}

// CHECK-LABEL: <fifo_runtime_set_address_invalid>:
// CHECK: ebreak

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
