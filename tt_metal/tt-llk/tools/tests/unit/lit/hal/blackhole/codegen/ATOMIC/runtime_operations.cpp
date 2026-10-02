// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

// Runtime fields are added to a constant operation word and pushed through the
// instruction buffer. Out-of-range runtime fields branch to an LLK assertion.

// ATINCGET 0x61: wrap 7 << 14 | word 1 << 12 | data 2 << 6 | address.
extern "C" __attribute__((noinline, used)) void runtime_fetch_increment_descriptor(std::uint32_t address)
{
    const auto descriptor = atomic::FetchIncrement {hal::gpr(address), hal::gpr<2>(), std::uint8_t {1}, std::uint8_t {8}};
    TT_INSN(descriptor.operation());
}

// CHECK-LABEL: <runtime_fetch_increment_descriptor>:
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
    const auto counter = atomic::counter<16>(hal::gpr<1>(), word);
    counter.fetch_add(hal::gpr<2>());
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
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    TT_INSN(fifo.operation().pop(hal::gpr<4>(), increment_log2));
}

// CHECK-LABEL: <fifo_runtime_increment>:
// CHECK-NEXT: li [[LIMIT:a[0-7]]],8
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
    const auto fifo = atomic::fifo<8>(hal::gpr(address));
    fifo.push(hal::gpr<4>());
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
    constexpr auto fifo = atomic::fifo<8>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr(result));
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
    lock.acquire();
    lock.release();
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
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set(compare, set);
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

// Pointer updates accept capacities up to 2^14; capacity 2^15 has a zero-width
// update mask.
extern "C" __attribute__((noinline, used)) void fifo_runtime_capacity(std::uint8_t capacity_log2)
{
    const auto descriptor = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, capacity_log2, std::uint8_t {0}, false};
    TT_INSN(descriptor.operation());
}

// CHECK-LABEL: <fifo_runtime_capacity>:
// CHECK: li [[LIMIT:a[0-7]]],14
// CHECK-NEXT: bltu [[LIMIT]],a0,[[FAIL:[0-9a-f]+]]
// CHECK: sw
// CHECK-NEXT: ret
// CHECK-EMPTY:
// CHECK-NEXT: [[FAIL]] <
// CHECK-NEXT: ebreak

// Both value-selected helpers must assert before issuing a capacity-15 update.
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
// CHECK-NEXT: ret

// The same capacity remains supported for read and write readiness probes.
extern "C" __attribute__((noinline, used)) void fifo_capacity15_waits()
{
    constexpr auto fifo = atomic::fifo<15>(hal::gpr<3>());
    fifo.wait_while<atomic::FifoState::Empty>(hal::gpr<4>());
    fifo.wait_while<atomic::FifoState::Full>(hal::gpr<4>());
}

// CHECK-LABEL: <fifo_capacity15_waits>:
// CHECK-NOT: ebreak
// CHECK: ttatincgetptr
// CHECK-NOT: ebreak
// CHECK: ttatincgetptr
// CHECK-NOT: ebreak
// CHECK: ret

// Static-looking values on the object execution path retain assertion-controlled
// validation; no new type-level rejection is required for these invalid operands.
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
