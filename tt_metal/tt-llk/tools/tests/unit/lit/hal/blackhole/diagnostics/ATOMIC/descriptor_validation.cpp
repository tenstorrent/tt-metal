// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -fsyntax-only %s

#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Counter<32, 1> counter_word3 {hal::gpr<1>(), 3};
constexpr atomic::Fifo<15, 3> fifo32k {hal::gpr<3>()};
constexpr auto lock_word3 = atomic::Lock {hal::gpr<63>(), 3};

static_assert(atomic::is_valid(counter_word3));
static_assert(atomic::is_valid(atomic::Counter<1, 63> {hal::gpr<63>()}));
static_assert(atomic::is_valid(fifo32k));
static_assert(atomic::is_valid(lock_word3));
static_assert(atomic::is_valid(counter_word3.fetch_add(hal::gpr<2>())));
static_assert(atomic::is_valid(fifo32k.pop_slots<15>(hal::gpr<4>())));
static_assert(atomic::is_valid(fifo32k.wait_while<atomic::FifoState::Full>(hal::gpr<4>())));
static_assert(atomic::is_valid(lock_word3.compare_set<15, 15>()));
static_assert(atomic::is_valid(atomic::MaskedWrite {hal::gpr<63>(), hal::gpr<60>(), std::uint8_t {0xff}}));

static_assert(!atomic::is_valid(atomic::Counter<0, 0> {hal::gpr<0>()}));
static_assert(!atomic::is_valid(atomic::Counter<33, 0> {hal::gpr<0>()}));
static_assert(!atomic::is_valid(atomic::Counter<32, 0> {hal::gpr<0>(), 4}));
static_assert(!atomic::is_valid(atomic::Counter<32, 64> {hal::gpr<64>()}));
static_assert(!atomic::is_valid(atomic::Fifo<16, 0> {hal::gpr<0>()}));
static_assert(!atomic::is_valid(atomic::Fifo<1, 64> {hal::gpr<64>()}));
static_assert(!atomic::is_valid(atomic::Lock {hal::gpr<0>(), 4}));
static_assert(!atomic::is_valid(atomic::Lock {hal::gpr<64>()}));

static_assert(!atomic::is_valid(atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<64>(), std::uint8_t {0}, std::uint8_t {32}}));
static_assert(!atomic::is_valid(atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<0>(), std::uint8_t {4}, std::uint8_t {32}}));
static_assert(!atomic::is_valid(atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<0>(), std::uint8_t {0}, std::uint8_t {0}}));
static_assert(!atomic::is_valid(atomic::FetchIncrement {hal::gpr<0>(), hal::gpr<0>(), std::uint8_t {0}, std::uint8_t {33}}));
static_assert(!atomic::is_valid(atomic::FifoAcquire {
    hal::gpr<0>(), hal::gpr<0>(), static_cast<atomic::FifoPointer>(2), std::uint8_t {0}, std::uint8_t {0}, false}));
static_assert(!atomic::is_valid(atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), atomic::FifoPointer::Read, std::uint8_t {16}, std::uint8_t {0}, false}));
static_assert(!atomic::is_valid(atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), atomic::FifoPointer::Read, std::uint8_t {0}, std::uint8_t {16}, false}));
// A readiness probe must not also advance the pointer.
static_assert(!atomic::is_valid(atomic::FifoAcquire {hal::gpr<0>(), hal::gpr<0>(), atomic::FifoPointer::Read, std::uint8_t {0}, std::uint8_t {1}, true}));
static_assert(!atomic::is_valid(atomic::CompareSet {hal::gpr<0>(), std::uint8_t {4}, std::uint8_t {0}, std::uint8_t {0}}));
static_assert(!atomic::is_valid(atomic::CompareSet {hal::gpr<0>(), std::uint8_t {0}, std::uint8_t {16}, std::uint8_t {0}}));
static_assert(!atomic::is_valid(atomic::CompareSet {hal::gpr<0>(), std::uint8_t {0}, std::uint8_t {0}, std::uint8_t {16}}));
static_assert(!atomic::is_valid(atomic::MaskedWrite {hal::gpr<0>(), hal::gpr<1>(), std::uint8_t {0xff}}));
static_assert(!atomic::is_valid(atomic::MaskedWrite {hal::gpr<64>(), hal::gpr<0>(), std::uint8_t {0xff}}));
