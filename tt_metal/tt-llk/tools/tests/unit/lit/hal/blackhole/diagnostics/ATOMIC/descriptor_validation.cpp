// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fsyntax-only %s

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Counter<32, 1> counter_word3 {hal::gpr<1>(), 3};
constexpr atomic::Counter<1, 63> counter_bit {hal::gpr<63>()};
constexpr auto fifo16k    = atomic::fifo<14>(hal::gpr<3>());
constexpr auto fifo32k    = atomic::fifo<15>(hal::gpr<3>());
constexpr auto lock_word3 = atomic::Lock {hal::gpr<63>(), 3};

static_assert(atomic::is_valid(counter_word3));
static_assert(atomic::is_valid(counter_bit));
static_assert(atomic::is_valid(fifo32k));
static_assert(atomic::is_valid(lock_word3));
static_assert(counter_word3.operation().fetch_add(hal::gpr<2>()) == TT_OP_ATINCGET(0, 31, 3, 2, 1));
static_assert(fifo16k.operation().pop(hal::gpr<4>()) == TT_OP_ATINCGETPTR(0, 0, 0, 15, 0, 4, 3));
static_assert(fifo16k.operation().push<14>(hal::gpr<4>()) == TT_OP_ATINCGETPTR(0, 0, 14, 15, 1, 4, 3));
static_assert(fifo32k.operation().wait_while<atomic::FifoState::Empty>(hal::gpr<4>()) == TT_OP_ATINCGETPTR(0, 1, 0, 0, 0, 4, 3));
static_assert(fifo32k.operation().wait_while<atomic::FifoState::Full>(hal::gpr<4>()) == TT_OP_ATINCGETPTR(0, 1, 0, 0, 1, 4, 3));
static_assert(lock_word3.operation().compare_set<15, 15>() == TT_OP_ATCAS(0, 15, 15, 3, 0, 63));
// Invalid raw descriptors remain constructible for explicit validation.
constexpr auto read_capacity15 = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, 15};
static_assert(!atomic::is_valid(read_capacity15));

constexpr auto write_capacity15 = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Write, 15, 15};
static_assert(!atomic::is_valid(write_capacity15));

constexpr auto increment_exceeds_capacity = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, 2, 3};
static_assert(!atomic::is_valid(increment_exceeds_capacity));

constexpr auto write_last_quad = atomic::MaskedWrite {hal::gpr<63>(), hal::gpr<60>(), 0xff};
static_assert(atomic::is_valid(write_last_quad));

constexpr atomic::Counter<0, 0> counter_zero_width {hal::gpr<0>()};
static_assert(!atomic::is_valid(counter_zero_width));

constexpr atomic::Counter<33, 0> counter_excess_width {hal::gpr<0>()};
static_assert(!atomic::is_valid(counter_excess_width));

constexpr atomic::Counter<32, 0> counter_invalid_word {hal::gpr<0>(), 4};
static_assert(!atomic::is_valid(counter_invalid_word));

constexpr atomic::Counter<32, 64> counter_invalid_address {hal::gpr<64>()};
static_assert(!atomic::is_valid(counter_invalid_address));

constexpr auto fifo_invalid_capacity = atomic::fifo<16>(hal::gpr<0>());
static_assert(!atomic::is_valid(fifo_invalid_capacity));

constexpr auto fifo_invalid_address = atomic::fifo<1>(hal::gpr<64>());
static_assert(!atomic::is_valid(fifo_invalid_address));

constexpr auto lock_invalid_word = atomic::Lock {hal::gpr<0>(), 4};
static_assert(!atomic::is_valid(lock_invalid_word));

constexpr auto lock_invalid_address = atomic::Lock {hal::gpr<64>()};
static_assert(!atomic::is_valid(lock_invalid_address));

constexpr auto fetch_invalid_data = atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<64>(), 0, 32};
static_assert(!atomic::is_valid(fetch_invalid_data));

constexpr auto fetch_invalid_word = atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<0>(), 4, 32};
static_assert(!atomic::is_valid(fetch_invalid_word));

constexpr auto fetch_zero_width = atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<0>(), 0, 0};
static_assert(!atomic::is_valid(fetch_zero_width));

constexpr auto fetch_excess_width = atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<0>(), 0, 33};
static_assert(!atomic::is_valid(fetch_excess_width));

constexpr auto acquire_invalid_pointer = atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), static_cast<atomic::FifoPointer>(2), 0, 0, false};
static_assert(!atomic::is_valid(acquire_invalid_pointer));

constexpr auto acquire_invalid_capacity = atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), atomic::FifoPointer::Read, 16, 0, false};
static_assert(!atomic::is_valid(acquire_invalid_capacity));

constexpr auto acquire_invalid_increment = atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), atomic::FifoPointer::Read, 0, 16, false};
static_assert(!atomic::is_valid(acquire_invalid_increment));

// A readiness probe must not also advance the pointer.
constexpr auto probe_with_increment = atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), atomic::FifoPointer::Read, 0, 1, true};
static_assert(!atomic::is_valid(probe_with_increment));

constexpr auto compare_set_invalid_word = atomic::CompareSet {hal::gpr<0>(), 0, 0, 4};
static_assert(!atomic::is_valid(compare_set_invalid_word));

constexpr auto compare_set_invalid_set = atomic::CompareSet {hal::gpr<0>(), 0, 16, 0};
static_assert(!atomic::is_valid(compare_set_invalid_set));

constexpr auto compare_set_invalid_compare = atomic::CompareSet {hal::gpr<0>(), 16, 0, 0};
static_assert(!atomic::is_valid(compare_set_invalid_compare));

constexpr auto write_unaligned_quad = atomic::MaskedWrite {hal::gpr<0>(), hal::gpr<1>(), 0xff};
static_assert(!atomic::is_valid(write_unaligned_quad));

constexpr auto write_invalid_address = atomic::MaskedWrite {hal::gpr<64>(), hal::gpr<0>(), 0xff};
static_assert(!atomic::is_valid(write_invalid_address));

// Descriptor defaults work with C++17 deduction; compare/set remain required.
constexpr auto default_fetch = atomic::FetchIncrement {hal::gpr<1>(), hal::gpr<2>()};
static_assert(default_fetch.word_select == 0 && default_fetch.counter_width == 32);
constexpr auto default_fifo = atomic::FifoAcquire {hal::gpr<3>(), hal::gpr<4>(), atomic::FifoPointer::Read, 3};
static_assert(default_fifo.increment_log2 == 0 && !default_fifo.probe_only);
constexpr auto default_cas = atomic::CompareSet {hal::gpr<7>(), 1, 0};
static_assert(default_cas.word_select == 0 && default_cas.compare == 1 && default_cas.set == 0);
constexpr atomic::CompareSet<7> explicit_cas {hal::gpr<7>(), 1, 0};
static_assert(explicit_cas.operation() == TT_OP_ATCAS(0, 0, 1, 0, 0, 7));
constexpr auto runtime_counter = atomic::counter<16>(hal::gpr(5), 2);
static_assert(atomic::is_valid(runtime_counter));
static_assert(runtime_counter.operation().fetch_add(hal::gpr(6)) == TT_OP_ATINCGET(0, 15, 2, 6, 5));
