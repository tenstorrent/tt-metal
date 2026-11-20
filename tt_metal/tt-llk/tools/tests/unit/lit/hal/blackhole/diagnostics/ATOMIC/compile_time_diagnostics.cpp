// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/zero-counter-width.cpp 2>&1 | FileCheck %s --check-prefix=ZERO_COUNTER_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/zero-counter-width.cpp 2>&1 | FileCheck %s --check-prefix=ZERO_COUNTER_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/invalid-counter-result.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/invalid-counter-result.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/invalid-counter-width.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/invalid-counter-width.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/invalid-counter-word.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_WORD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/invalid-counter-word.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_WORD
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/invalid-fifo.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_FIFO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/invalid-fifo.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_FIFO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/fifo-increment-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=FIFO_INCREMENT_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/fifo-increment-exceeds-capacity.cpp 2>&1 | FileCheck %s --check-prefix=FIFO_INCREMENT_EXCEEDS_CAPACITY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/fifo-runtime-increment-constant.cpp 2>&1 | FileCheck %s --check-prefix=FIFO_RUNTIME_INCREMENT_CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/fifo-runtime-increment-constant.cpp 2>&1 | FileCheck %s --check-prefix=FIFO_RUNTIME_INCREMENT_CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/invalid-lock.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_LOCK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/invalid-lock.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_LOCK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/lock-compare-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=LOCK_COMPARE_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/lock-set-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=LOCK_SET_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/missing-cas-set.cpp 2>&1 | FileCheck %s --check-prefix=MISSING_CAS_SET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/misaligned-masked-write.cpp 2>&1 | FileCheck %s --check-prefix=MISALIGNED_MASKED_WRITE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/runtime-masked-write-address.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_MASKED_WRITE_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/runtime-masked-write-data.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_MASKED_WRITE_DATA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/capacity15-pop.cpp 2>&1 | FileCheck %s --check-prefix=CAPACITY15_POP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/capacity15-pop.cpp 2>&1 | FileCheck %s --check-prefix=CAPACITY15_POP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/capacity15-push.cpp 2>&1 | FileCheck %s --check-prefix=CAPACITY15_PUSH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -DENABLE_LLK_ASSERT %t/capacity15-push.cpp 2>&1 | FileCheck %s --check-prefix=CAPACITY15_PUSH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-counter-width.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_COUNTER_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-counter-address.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_COUNTER_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-fifo-address.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_FIFO_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-fifo-result.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_FIFO_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-wait-capacity.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_WAIT_CAPACITY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-wait-address.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_WAIT_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-wait-result.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_WAIT_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-lock-address.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_LOCK_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-masked-write-address.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_MASKED_WRITE_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-masked-write-data.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_MASKED_WRITE_DATA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} -UENABLE_LLK_ASSERT %t/issued-masked-write-quad.cpp 2>&1 | FileCheck %s --check-prefix=ISSUED_MASKED_WRITE_QUAD
// clang-format on

//--- zero-counter-width.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto descriptor = atomic::FetchIncrement {hal::gpr<1>(), hal::gpr<2>(), 0, 0};
constexpr auto bad        = descriptor.operation();

// ZERO_COUNTER_WIDTH: in 'constexpr' expansion of 'hal::detail::require({{.*}}"ATINCGET counter width must be in [1, 32]"
// ZERO_COUNTER_WIDTH: error: '__builtin_trap()' is not a constant expression

//--- invalid-counter-result.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto counter = atomic::counter<32>(hal::gpr<1>());
constexpr auto bad     = counter.operation().fetch_add(hal::gpr<64>());

// INVALID_COUNTER_RESULT: error: static assertion failed: ATINCGET data GPR index must be in [0, 63]

//--- invalid-counter-width.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto counter = atomic::counter<33>(hal::gpr<1>());
constexpr auto bad     = counter.operation().fetch_add(hal::gpr<2>());

// INVALID_COUNTER_WIDTH: error: static assertion failed: ATINCGET counter width must be in [1, 32]

//--- invalid-counter-word.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto counter = atomic::counter<32>(hal::gpr<1>(), 4 /*word*/);
constexpr auto bad     = counter.operation().fetch_add(hal::gpr<2>());

// INVALID_COUNTER_WORD: in 'constexpr' expansion of 'hal::detail::require({{.*}}"ATINCGET word selection must be in [0, 3]"
// INVALID_COUNTER_WORD: error: '__builtin_trap()' is not a constant expression

//--- invalid-fifo.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto fifo = atomic::fifo<16>(hal::gpr<3>());
constexpr auto bad  = fifo.operation().pop(hal::gpr<4>());

// INVALID_FIFO: error: static assertion failed: ATINCGETPTR FIFO capacity logarithm must be in [0, 14] for pointer updates

//--- fifo-increment-too-wide.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f()
{
    constexpr auto fifo = atomic::fifo<3>(hal::gpr<3>());
    fifo.push<16>(hal::gpr<4>());
}

// FIFO_INCREMENT_TOO_WIDE: static assertion failed: FIFO increment logarithm must be in [0, 15]

//--- fifo-increment-exceeds-capacity.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f()
{
    constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
    fifo.pop<3>(hal::gpr<4>());
}

// FIFO_INCREMENT_EXCEEDS_CAPACITY: static assertion failed: FIFO increment logarithm must not exceed the FIFO capacity logarithm

//--- fifo-runtime-increment-constant.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto fifo = atomic::fifo<2>(hal::gpr<3>());
constexpr auto bad  = fifo.operation().pop(hal::gpr<4>(), 3);

// FIFO_RUNTIME_INCREMENT_CONSTANT: in 'constexpr' expansion of 'hal::detail::require({{.*}}"ATINCGETPTR increment logarithm must not exceed
// FIFO_RUNTIME_INCREMENT_CONSTANT: error: '__builtin_trap()' is not a constant expression

//--- invalid-lock.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto lock = atomic::Lock {hal::gpr<7>(), 4 /*word*/};
constexpr auto bad  = lock.operation().acquire();

// INVALID_LOCK: in 'constexpr' expansion of 'hal::detail::require({{.*}}"ATCAS word selection must be in [0, 3]"
// INVALID_LOCK: error: '__builtin_trap()' is not a constant expression

//--- lock-compare-too-wide.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f()
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set<16, 0>();
}

// LOCK_COMPARE_TOO_WIDE: static assertion failed: ATCAS compare value must fit in four bits

//--- lock-set-too-wide.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f()
{
    constexpr auto lock = atomic::Lock {hal::gpr<7>()};
    lock.compare_set<0, 16>();
}

// LOCK_SET_TOO_WIDE: static assertion failed: ATCAS replacement value must fit in four bits

//--- missing-cas-set.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr atomic::CompareSet<7> bad {hal::gpr<7>(), 1};

// MISSING_CAS_SET: no matching function

//--- misaligned-masked-write.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f()
{
    atomic::store_masked<0xff>(hal::gpr<8>(), hal::gpr<13>());
}

// MISALIGNED_MASKED_WRITE: static assertion failed: invalid atomic descriptor

//--- runtime-masked-write-address.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f(unsigned index)
{
    atomic::store_masked<0xff>(hal::gpr(index), hal::gpr<12>());
}

// RUNTIME_MASKED_WRITE_ADDRESS: static assertion failed: immediate masked write requires hal::gpr<Index>() for its address

//--- runtime-masked-write-data.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void f(unsigned index)
{
    atomic::store_masked<0xff>(hal::gpr<8>(), hal::gpr(index));
}

// RUNTIME_MASKED_WRITE_DATA: static assertion failed: immediate masked write requires hal::gpr<Index>() for its data quad

//--- capacity15-pop.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto fifo = atomic::fifo<15>(hal::gpr<3>());
constexpr auto bad  = fifo.operation().pop(hal::gpr<4>());

// CAPACITY15_POP: error: static assertion failed: ATINCGETPTR FIFO capacity logarithm must be in [0, 14] for pointer updates

//--- capacity15-push.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto fifo = atomic::fifo<15>(hal::gpr<3>());
constexpr auto bad  = fifo.operation().push(hal::gpr<4>());

// CAPACITY15_PUSH: error: static assertion failed: ATINCGETPTR FIFO capacity logarithm must be in [0, 14] for pointer updates

// Issuing calls reject invalid template operands at compile time, in builds with and without LLK assertions.

//--- issued-counter-width.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_counter_width()
{
    atomic::counter<0>(hal::gpr<1>()).fetch_add(hal::gpr<2>());
}

// ISSUED_COUNTER_WIDTH: error: static assertion failed: ATINCGET counter width must be in [1, 32]

//--- issued-counter-address.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_counter_address()
{
    atomic::counter<32>(hal::gpr<64>()).fetch_add(hal::gpr<2>());
}

// ISSUED_COUNTER_ADDRESS: error: static assertion failed: ATINCGET address GPR index must be in [0, 63]

//--- issued-fifo-address.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_fifo_address()
{
    atomic::fifo<3>(hal::gpr<64>()).pop(hal::gpr<4>());
}

// ISSUED_FIFO_ADDRESS: error: static assertion failed: ATINCGETPTR address GPR index must be in [0, 63]

//--- issued-fifo-result.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_fifo_result()
{
    atomic::fifo<3>(hal::gpr<3>()).push(hal::gpr<64>());
}

// ISSUED_FIFO_RESULT: error: static assertion failed: ATINCGETPTR result GPR index must be in [0, 63]

//--- issued-wait-capacity.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_wait_capacity()
{
    atomic::fifo<16>(hal::gpr<3>()).wait_while<atomic::FifoState::Empty>(hal::gpr<4>());
}

// ISSUED_WAIT_CAPACITY: error: static assertion failed: ATINCGETPTR FIFO capacity logarithm must be in [0, 15] for readiness probes

//--- issued-wait-address.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_wait_address()
{
    atomic::fifo<3>(hal::gpr<64>()).wait_while<atomic::FifoState::Full>(hal::gpr<4>());
}

// ISSUED_WAIT_ADDRESS: error: static assertion failed: ATINCGETPTR address GPR index must be in [0, 63]

//--- issued-wait-result.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_wait_result()
{
    atomic::fifo<3>(hal::gpr<3>()).wait_while<atomic::FifoState::Full>(hal::gpr<64>());
}

// ISSUED_WAIT_RESULT: error: static assertion failed: ATINCGETPTR result GPR index must be in [0, 63]

//--- issued-lock-address.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_lock_address()
{
    atomic::Lock {hal::gpr<64>()}.acquire();
}

// ISSUED_LOCK_ADDRESS: error: static assertion failed: ATCAS address GPR index must be in [0, 63]

//--- issued-masked-write-address.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_masked_write_address()
{
    atomic::store_masked(hal::gpr<64>(), hal::gpr<4>(), 0xff /*granule_mask*/);
}

// ISSUED_MASKED_WRITE_ADDRESS: error: static assertion failed: ATSWAP address GPR index must be in [0, 63]

//--- issued-masked-write-data.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_masked_write_data()
{
    atomic::store_masked(hal::gpr<8>(), hal::gpr<64>(), 0xff /*granule_mask*/);
}

// ISSUED_MASKED_WRITE_DATA: error: static assertion failed: ATSWAP data GPR index must be in [0, 63]

//--- issued-masked-write-quad.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

void issued_masked_write_quad()
{
    atomic::store_masked(hal::gpr<3>(), hal::gpr<5>(), 0xff /*granule_mask*/);
}

// ISSUED_MASKED_WRITE_QUAD: error: static assertion failed: ATSWAP data GPR must be four-aligned
