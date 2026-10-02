// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

// Wrap fields are encoded as width - 1 for ATINCGET and capacity_log2 + 1 (mod 16)
// for ATINCGETPTR (zero is supported only for readiness probes).
// ATCAS prints the set value before the compare value.
extern "C" __attribute__((noinline, used)) void descriptor_operations()
{
    constexpr auto fetch       = atomic::FetchIncrement {hal::gpr<1>(), hal::gpr<2>(), std::uint8_t {3}, std::uint8_t {16}};
    constexpr auto acquire     = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Write, std::uint8_t {3}, std::uint8_t {2}, false};
    constexpr auto compare_set = atomic::CompareSet {hal::gpr<7>(), std::uint8_t {5}, std::uint8_t {9}, std::uint8_t {2}};
    constexpr auto write       = atomic::MaskedWrite {hal::gpr<8>(), hal::gpr<12>(), std::uint8_t {0xa5}};
    TTI_INSN(fetch.operation());
    TTI_INSN(acquire.operation());
    TTI_INSN(compare_set.operation());
    TTI_INSN(write.operation());
}

// CHECK-LABEL: <descriptor_operations>:
// CHECK-NEXT: ttatincget 0,15,3,2,1
// CHECK-NEXT: ttatincgetptr 0,0,2,4,1,4,3
// CHECK-NEXT: ttatcas 0,9,5,2,0,7
// CHECK-NEXT: ttatswap 0,165,12,8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void counter_fetch_add()
{
    constexpr auto counter_word3 = atomic::counter<32>(hal::gpr<1>(), 3);
    constexpr atomic::Counter<1, 63> counter_bit {hal::gpr<63>()};
    counter_word3.fetch_add(hal::gpr<2>());
    counter_bit.fetch_add(hal::gpr<63>());
}

// CHECK-LABEL: <counter_fetch_add>:
// CHECK-NEXT: ttatincget 0,31,3,2,1
// CHECK-NEXT: ttatincget 0,0,0,63,63
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void fifo_pop_push()
{
    constexpr auto fifo8   = atomic::fifo<3>(hal::gpr<3>());
    constexpr auto fifo16k = atomic::fifo<14>(hal::gpr<5>());
    fifo8.pop(hal::gpr<4>());
    fifo8.pop<2>(hal::gpr<4>());
    fifo8.push(hal::gpr<6>());
    fifo8.push<2>(hal::gpr<6>());
    fifo16k.pop<14>(hal::gpr<0>());
    fifo16k.push<14>(hal::gpr<0>());
}

// CHECK-LABEL: <fifo_pop_push>:
// CHECK-NEXT: ttatincgetptr 0,0,0,4,0,4,3
// CHECK-NEXT: ttatincgetptr 0,0,2,4,0,4,3
// CHECK-NEXT: ttatincgetptr 0,0,0,4,1,6,3
// CHECK-NEXT: ttatincgetptr 0,0,2,4,1,6,3
// CHECK-NEXT: ttatincgetptr 0,0,14,15,0,0,5
// CHECK-NEXT: ttatincgetptr 0,0,14,15,1,0,5
// CHECK-NEXT: ret

// Waits set NoIncr and select the read pointer for Empty and the write pointer for Full.
extern "C" __attribute__((noinline, used)) void fifo_wait_while()
{
    constexpr auto fifo1   = atomic::fifo<0>(hal::gpr<0>());
    constexpr auto fifo32k = atomic::fifo<15>(hal::gpr<5>());
    fifo1.wait_while<atomic::FifoState::Empty>(hal::gpr<0>());
    fifo32k.wait_while<atomic::FifoState::Empty>(hal::gpr<6>());
    fifo32k.wait_while<atomic::FifoState::Full>(hal::gpr<6>());
}

// CHECK-LABEL: <fifo_wait_while>:
// CHECK-NEXT: ttatincgetptr 0,1,0,1,0,0,0
// CHECK-NEXT: ttatincgetptr 0,1,0,0,0,6,5
// CHECK-NEXT: ttatincgetptr 0,1,0,0,1,6,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void lock_operations()
{
    constexpr auto lock_word2 = atomic::Lock {hal::gpr<7>(), 2};
    constexpr auto lock_word0 = atomic::Lock {hal::gpr<63>()};
    lock_word2.acquire();
    lock_word2.release();
    lock_word2.compare_set<15, 3>();
    lock_word0.acquire();
}

// CHECK-LABEL: <lock_operations>:
// CHECK-NEXT: ttatcas 0,1,0,2,0,7
// CHECK-NEXT: ttatcas 0,0,1,2,0,7
// CHECK-NEXT: ttatcas 0,3,15,2,0,7
// CHECK-NEXT: ttatcas 0,1,0,0,0,63
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void masked_store()
{
    atomic::store_masked<0xa5>(hal::gpr<8>(), hal::gpr<12>());
    atomic::store_masked<0xff>(hal::gpr<63>(), hal::gpr<60>());
}

// CHECK-LABEL: <masked_store>:
// CHECK-NEXT: ttatswap 0,165,12,8
// CHECK-NEXT: ttatswap 0,255,60,63
// CHECK-NEXT: ret

// MOP builders remain constant expressions and never issue instructions.
constexpr auto counter_word3  = atomic::counter<32>(hal::gpr<1>(), 3);
constexpr auto fifo8          = atomic::fifo<3>(hal::gpr<3>());
constexpr auto lock_word2     = atomic::Lock {hal::gpr<7>(), 2};
constexpr std::uint32_t mop[] = {
    counter_word3.operation().fetch_add(hal::gpr<2>()),
    fifo8.operation().pop<2>(hal::gpr<4>()),
    fifo8.operation().push(hal::gpr<6>()),
    fifo8.operation().wait_while<atomic::FifoState::Empty>(hal::gpr<4>()),
    lock_word2.operation().acquire(),
    lock_word2.operation().release(),
};
static_assert(mop[0] == TT_OP_ATINCGET(0, 31, 3, 2, 1));
static_assert(mop[1] == TT_OP_ATINCGETPTR(0, 0, 2, 4, 0, 4, 3));
static_assert(mop[2] == TT_OP_ATINCGETPTR(0, 0, 0, 4, 1, 6, 3));
static_assert(mop[3] == TT_OP_ATINCGETPTR(0, 1, 0, 4, 0, 4, 3));
static_assert(mop[4] == TT_OP_ATCAS(0, 1, 0, 2, 0, 7));
static_assert(mop[5] == TT_OP_ATCAS(0, 0, 1, 2, 0, 7));

// Callers pass byte addresses. L1_ADDRESS(0x1000) + 1 loads the atomic line address 0x100.
extern "C" __attribute__((noinline, used)) void fifo_address_setup()
{
    constexpr auto fifo = atomic::fifo<3>(hal::gpr<3>());
    fifo.set_address(0x1000);
}

// CHECK-LABEL: <fifo_address_setup>:
// CHECK-NEXT: ttsetdmareg 0,256,0,6
// CHECK-NEXT: ttsetdmareg 0,0,0,7
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void counter_and_lock_address_setup()
{
    constexpr auto counter = atomic::counter<32>(hal::gpr<1>(), 3);
    constexpr auto lock    = atomic::Lock {hal::gpr<7>(), 2};
    counter.set_address(0x1000);
    lock.set_address(0x2000);
}

// CHECK-LABEL: <counter_and_lock_address_setup>:
// CHECK-NEXT: ttsetdmareg 0,256,0,2
// CHECK-NEXT: ttsetdmareg 0,0,0,3
// CHECK-NEXT: ttsetdmareg 0,512,0,14
// CHECK-NEXT: ttsetdmareg 0,0,0,15
// CHECK-NEXT: ret
