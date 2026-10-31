// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// Execution uses TT_* references; MOP encoding uses TT_OP_* references.

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_fetch_increment_descriptor(const std::uint32_t address)
{
    const auto descriptor = atomic::FetchIncrement {hal::gpr(address), hal::gpr<2>(), std::uint8_t {1}, std::uint8_t {8}};
    return descriptor.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_runtime_fetch_increment_descriptor(const std::uint32_t address)
{
    return TT_OP_ATINCGET(0 /*MemHierSel*/, 7 /*WrapVal*/, 1 /*Sel32b*/, 2 /*DataRegIndex*/, address);
}

extern "C" __attribute__((noinline, used)) void counter_runtime_address(const std::uint32_t address)
{
    const auto counter = atomic::counter<8>(hal::gpr(address), 1 /*word*/);
    counter.fetch_add(hal::gpr<2>());
}

extern "C" __attribute__((noinline, used)) void reference_counter_runtime_address(const std::uint32_t address)
{
    TT_ATINCGET(0 /*MemHierSel*/, 7 /*WrapVal*/, 1 /*Sel32b*/, 2 /*DataRegIndex*/, address);
}

extern "C" __attribute__((noinline, used)) void counter_runtime_word(const std::uint8_t word)
{
    const auto counter = atomic::counter<16>(hal::gpr<1>(), word);
    counter.fetch_add(hal::gpr<2>());
}

extern "C" __attribute__((noinline, used)) void reference_counter_runtime_word(const std::uint8_t word)
{
    TT_ATINCGET(0 /*MemHierSel*/, 15 /*WrapVal*/, word, 2 /*DataRegIndex*/, 1 /*AddrRegIndex*/);
}

extern "C" __attribute__((noinline, used)) void fifo_runtime_increment(const std::uint8_t increment_log2)
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    fifo.pop(hal::gpr<4>(), increment_log2);
}

extern "C" __attribute__((noinline, used)) void reference_fifo_runtime_increment(const std::uint8_t increment_log2)
{
    TT_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, increment_log2, 9 /*WrapVal*/, 0 /*Sel32b*/, 4 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t encode_fifo_runtime_increment(const std::uint8_t increment_log2)
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    return fifo.operation().pop(hal::gpr<4>(), increment_log2);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_encode_fifo_runtime_increment(const std::uint8_t increment_log2)
{
    return TT_OP_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, increment_log2, 9 /*WrapVal*/, 0 /*Sel32b*/, 4 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}

extern "C" __attribute__((noinline, used)) void fifo_runtime_push_default(const std::uint32_t address)
{
    const auto fifo = atomic::fifo<8>(hal::gpr(address));
    fifo.push(hal::gpr<4>());
}

extern "C" __attribute__((noinline, used)) void reference_fifo_runtime_push_default(const std::uint32_t address)
{
    TT_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, 0 /*IncrVal*/, 9 /*WrapVal*/, 1 /*Sel32b*/, 4 /*DataRegIndex*/, address);
}

extern "C" __attribute__((noinline, used)) void fifo_runtime_wait_full(const std::uint32_t result)
{
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr(result));
}

extern "C" __attribute__((noinline, used)) void reference_fifo_runtime_wait_full(const std::uint32_t result)
{
    TT_ATINCGETPTR(0 /*MemHierSel*/, 1 /*NoIncr*/, 0 /*IncrVal*/, 9 /*WrapVal*/, 1 /*Sel32b*/, result, 3 /*AddrRegIndex*/);
}

extern "C" __attribute__((noinline, used)) void lock_runtime_acquire_release(const std::uint32_t address)
{
    const auto lock = atomic::Lock {hal::gpr(address), 1};
    lock.acquire();
    lock.release();
}

extern "C" __attribute__((noinline, used)) void reference_lock_runtime_acquire_release(const std::uint32_t address)
{
    TT_ATCAS(0 /*MemHierSel*/, 1 /*SwapVal*/, 0 /*CmpVal*/, 1 /*Sel32b*/, 0 /*DataRegIndex*/, address);
    TT_ATCAS(0 /*MemHierSel*/, 0 /*SwapVal*/, 1 /*CmpVal*/, 1 /*Sel32b*/, 0 /*DataRegIndex*/, address);
}

// Keep both fields runtime-selected while avoiding incidental reassociation
// differences between the HAL and TT_* wrappers taking two separate arguments.
struct CompareSetValues
{
    std::uint8_t compare;
    std::uint8_t set;
};

extern "C" __attribute__((noinline, used)) void lock_runtime_compare_set(const CompareSetValues values)
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(values.compare, values.set);
}

extern "C" __attribute__((noinline, used)) void reference_lock_runtime_compare_set(const CompareSetValues values)
{
    TT_ATCAS(0 /*MemHierSel*/, values.set, values.compare, 0 /*Sel32b*/, 0 /*DataRegIndex*/, 7 /*AddrRegIndex*/);
}

extern "C" __attribute__((noinline, used)) void masked_store_runtime_mask(const std::uint8_t mask)
{
    atomic::store_masked(hal::gpr<8>(), hal::gpr<12>(), mask);
}

extern "C" __attribute__((noinline, used)) void reference_masked_store_runtime_mask(const std::uint8_t mask)
{
    TT_ATSWAP(0 /*MemHierSel*/, mask, 12 /*DataRegIndex*/, 8 /*AddrRegIndex*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t fifo_runtime_capacity(const std::uint8_t capacity_log2)
{
    const auto descriptor = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, capacity_log2, std::uint8_t {0}, false};
    return descriptor.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_fifo_runtime_capacity(const std::uint8_t capacity_log2)
{
    return TT_OP_ATINCGETPTR(0 /*MemHierSel*/, 0 /*NoIncr*/, 0 /*IncrVal*/, (capacity_log2 + 1u) & 0xfu, 0 /*Sel32b*/, 4 /*DataRegIndex*/, 3 /*AddrRegIndex*/);
}
