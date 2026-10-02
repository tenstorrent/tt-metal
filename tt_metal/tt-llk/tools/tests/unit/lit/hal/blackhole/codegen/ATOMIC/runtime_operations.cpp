// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

// Runtime fields are added to a constant operation word and pushed through the
// instruction buffer. Out-of-range runtime fields branch to an LLK assertion.

template <std::uint32_t AddressIndex>
constexpr atomic::Counter<16, AddressIndex> counter16(const hal::Gpr<AddressIndex> address, const std::uint8_t word)
{
    return {address, word};
}

template <std::uint32_t AddressIndex>
constexpr atomic::Fifo<8, AddressIndex> fifo256(const hal::Gpr<AddressIndex> address)
{
    return {address};
}

// ATINCGET 0x61: wrap 7 << 14 | word 1 << 12 | data 2 << 6 | address.
extern "C" __attribute__((noinline, used)) void run_runtime_fetch_increment(std::uint32_t address)
{
    atomic::run(atomic::FetchIncrement {hal::gpr(address), hal::gpr<2>(), std::uint8_t {1}, std::uint8_t {8}});
}

// CHECK-LABEL: <run_runtime_fetch_increment>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[OP:a[0-7]]],0x6101d
// CHECK-NEXT: addi [[OP]],[[OP]],128
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ATINCGET 0x61: wrap 15 << 14 | word << 12 | data 2 << 6 | address 1.
extern "C" __attribute__((noinline, used)) void counter_runtime_word(std::uint8_t word)
{
    atomic::fetch_add(counter16(hal::gpr<1>(), word), hal::gpr<2>());
}

// CHECK-LABEL: <counter_runtime_word>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],3
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[BASE:a[0-7]]],0x6103c
// CHECK-NEXT: addi [[OP:a[0-7]]],[[BASE]],129
// CHECK-NEXT: slli a0,a0,0xc
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ATINCGETPTR 0x62: increment << 18 | wrap 9 << 14 | read | result 4 << 6 | address 3.
extern "C" __attribute__((noinline, used)) void fifo_runtime_increment(std::uint8_t increment_log2)
{
    atomic::pop_slots(fifo256(hal::gpr<3>()), hal::gpr<4>(), increment_log2);
}

// CHECK-LABEL: <fifo_runtime_increment>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],15
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[BASE:a[0-7]]],0x62024
// CHECK-NEXT: addi [[OP:a[0-7]]],[[BASE]],259
// CHECK-NEXT: slli a0,a0,0x12
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ATINCGETPTR 0x62: wrap 9 << 14 | write 1 << 12 | result 4 << 6 | address.
extern "C" __attribute__((noinline, used)) void fifo_runtime_push_default(std::uint32_t address)
{
    atomic::push_slots(fifo256(hal::gpr(address)), hal::gpr<4>());
}

// CHECK-LABEL: <fifo_runtime_push_default>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[OP:a[0-7]]],0x62025
// CHECK-NEXT: addi [[OP]],[[OP]],256
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ATINCGETPTR 0x62: no-increment 1 << 22 | wrap 9 << 14 | write 1 << 12 | result << 6 | address 3.
extern "C" __attribute__((noinline, used)) void fifo_runtime_wait_full(std::uint32_t result)
{
    atomic::wait_while<atomic::FifoState::Full>(fifo256(hal::gpr<3>()), hal::gpr(result));
}

// CHECK-LABEL: <fifo_runtime_wait_full>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: lui [[BASE:a[0-7]]],0x62425
// CHECK-NEXT: addi [[OP:a[0-7]]],[[BASE]],3
// CHECK-NEXT: slli a0,a0,0x6
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ATCAS 0x64: acquire is set 1 << 18 | compare 0 and release is compare 1 << 14 | set 0,
// both with word 1 << 12 | address.
extern "C" __attribute__((noinline, used)) void lock_runtime_acquire_release(std::uint32_t address)
{
    const auto lock = atomic::Lock {hal::gpr(address), 1};
    atomic::acquire(lock);
    atomic::release(lock);
}

// CHECK-LABEL: <lock_runtime_acquire_release>:
// CHECK-NEXT: lui [[ACQUIRE:a[0-7]]],0x64041
// CHECK-NEXT: li [[LIMIT:a[0-7]]],63
// CHECK-NEXT: add [[ACQUIRE]],a0,[[ACQUIRE]]
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: sw [[ACQUIRE]],0([[BUF:a[0-7]]])
// CHECK-NEXT: lui [[RELEASE:a[0-7]]],0x64005
// CHECK-NEXT: add a0,a0,[[RELEASE]]
// CHECK-NEXT: sw a0,0([[BUF]])
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// ATCAS 0x64: set << 18 | compare << 14 | address 7, each value checked against four bits.
extern "C" __attribute__((noinline, used)) void lock_runtime_compare_set(std::uint8_t compare, std::uint8_t set)
{
    atomic::compare_set(atomic::Lock {hal::gpr<7>()}, compare, set);
}

// CHECK-LABEL: <lock_runtime_compare_set>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],15
// CHECK-NEXT: bltu [[LIMIT]],a0,
// CHECK-NEXT: R_RISCV_BRANCH
// CHECK-NEXT: li [[LIMIT]],15
// CHECK-NEXT: bltu [[LIMIT]],a1,
// CHECK: lui [[BASE:a[0-7]]],0x64000
// CHECK-NEXT: addi [[BASE]],[[BASE]],7
// CHECK-NEXT: slli a1,a1,0x12
// CHECK-NEXT: add a1,a1,[[BASE]]
// CHECK-NEXT: slli a0,a0,0xe
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a1,a1,a0
// CHECK: sw a1,0(
// CHECK-NEXT: ret
// CHECK: ebreak
// CHECK: ebreak

// ATSWAP 0x63: mask << 14 | data 12 << 6 | address 8. An eight-bit mask always fits.
extern "C" __attribute__((noinline, used)) void masked_store_runtime_mask(std::uint8_t mask)
{
    atomic::store_masked(hal::gpr<8>(), hal::gpr<12>(), mask);
}

// CHECK-LABEL: <masked_store_runtime_mask>:
// CHECK-NEXT: lui [[BASE:a[0-7]]],0x63000
// CHECK-NEXT: addi [[OP:a[0-7]]],[[BASE]],776
// CHECK-NEXT: slli a0,a0,0xe
// CHECK: R_RISCV_HI20 __instrn_buffer
// CHECK: add a0,a0,[[OP]]
// CHECK: sw a0,0(
// CHECK-NEXT: ret
// CHECK-NOT: ebreak
