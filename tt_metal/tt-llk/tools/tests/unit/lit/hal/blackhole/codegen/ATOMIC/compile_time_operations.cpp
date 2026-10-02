// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Counter<32, 1> counter_word3 {hal::gpr<1>(), 3};
constexpr atomic::Counter<1, 63> counter_bit {hal::gpr<63>()};
constexpr atomic::Fifo<3, 3> fifo8 {hal::gpr<3>()};
constexpr atomic::Fifo<0, 0> fifo1 {hal::gpr<0>()};
constexpr atomic::Fifo<15, 5> fifo32k {hal::gpr<5>()};
constexpr auto lock_word2 = atomic::Lock {hal::gpr<7>(), 2};
constexpr auto lock_word0 = atomic::Lock {hal::gpr<63>()};

// Wrap fields are encoded as width - 1 for ATINCGET and capacity_log2 + 1 (mod 16)
// for ATINCGETPTR; ATCAS prints the set value before the compare value.
extern "C" __attribute__((noinline, used)) void run_descriptors()
{
    atomic::run<atomic::FetchIncrement {hal::gpr<1>(), hal::gpr<2>(), std::uint8_t {3}, std::uint8_t {16}}>();
    atomic::run<atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Write, std::uint8_t {3}, std::uint8_t {2}, false}>();
    atomic::run<atomic::CompareSet {hal::gpr<7>(), std::uint8_t {2}, std::uint8_t {5}, std::uint8_t {9}}>();
    atomic::run<atomic::MaskedWrite {hal::gpr<8>(), hal::gpr<12>(), std::uint8_t {0xa5}}>();
}

// CHECK-LABEL: <run_descriptors>:
// CHECK-NEXT: ttatincget 0,15,3,2,1
// CHECK-NEXT: ttatincgetptr 0,0,2,4,1,4,3
// CHECK-NEXT: ttatcas 0,9,5,2,0,7
// CHECK-NEXT: ttatswap 0,165,12,8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void counter_fetch_add()
{
    atomic::fetch_add<counter_word3>(hal::gpr<2>());
    atomic::fetch_add<counter_bit>(hal::gpr<63>());
}

// CHECK-LABEL: <counter_fetch_add>:
// CHECK-NEXT: ttatincget 0,31,3,2,1
// CHECK-NEXT: ttatincget 0,0,0,63,63
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void fifo_pop_push()
{
    atomic::pop_slots<fifo8>(hal::gpr<4>());
    atomic::pop_slots<fifo8, 2>(hal::gpr<4>());
    atomic::push_slots<fifo8>(hal::gpr<6>());
    atomic::push_slots<fifo8, 2>(hal::gpr<6>());
    atomic::pop_slots<fifo32k, 15>(hal::gpr<0>());
}

// CHECK-LABEL: <fifo_pop_push>:
// CHECK-NEXT: ttatincgetptr 0,0,0,4,0,4,3
// CHECK-NEXT: ttatincgetptr 0,0,2,4,0,4,3
// CHECK-NEXT: ttatincgetptr 0,0,0,4,1,6,3
// CHECK-NEXT: ttatincgetptr 0,0,2,4,1,6,3
// CHECK-NEXT: ttatincgetptr 0,0,15,0,0,0,5
// CHECK-NEXT: ret

// Waits set NoIncr and select the read pointer for Empty and the write pointer for Full.
extern "C" __attribute__((noinline, used)) void fifo_wait_while()
{
    atomic::wait_while<atomic::FifoState::Empty, fifo1>(hal::gpr<0>());
    atomic::wait_while<atomic::FifoState::Full, fifo32k>(hal::gpr<6>());
}

// CHECK-LABEL: <fifo_wait_while>:
// CHECK-NEXT: ttatincgetptr 0,1,0,1,0,0,0
// CHECK-NEXT: ttatincgetptr 0,1,0,0,1,6,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void lock_operations()
{
    atomic::acquire<lock_word2>();
    atomic::release<lock_word2>();
    atomic::compare_set<lock_word2, 15, 3>();
    atomic::acquire<lock_word0>();
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
