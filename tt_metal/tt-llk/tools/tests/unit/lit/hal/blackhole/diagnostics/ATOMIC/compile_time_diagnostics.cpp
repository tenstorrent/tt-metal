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
// clang-format on

//--- zero-counter-width.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto descriptor = atomic::FetchIncrement {hal::gpr<1>(), hal::gpr<2>(), 0, 0};
constexpr auto bad        = descriptor.operation();

// ZERO_COUNTER_WIDTH: not a constant expression

//--- invalid-counter-result.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto counter = atomic::counter<32>(hal::gpr<1>());
constexpr auto bad     = counter.operation().fetch_add(hal::gpr<64>());

// INVALID_COUNTER_RESULT: not a constant expression

//--- invalid-counter-width.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto counter = atomic::counter<33>(hal::gpr<1>());
constexpr auto bad     = counter.operation().fetch_add(hal::gpr<2>());

// INVALID_COUNTER_WIDTH: not a constant expression

//--- invalid-counter-word.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto counter = atomic::counter<32>(hal::gpr<1>(), 4);
constexpr auto bad     = counter.operation().fetch_add(hal::gpr<2>());

// INVALID_COUNTER_WORD: not a constant expression

//--- invalid-fifo.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto fifo = atomic::fifo<16>(hal::gpr<3>());
constexpr auto bad  = fifo.operation().pop(hal::gpr<4>());

// INVALID_FIFO: not a constant expression

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

// FIFO_RUNTIME_INCREMENT_CONSTANT: not a constant expression

//--- invalid-lock.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto lock = atomic::Lock {hal::gpr<7>(), 4};
constexpr auto bad  = lock.operation().acquire();

// INVALID_LOCK: not a constant expression

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

// CAPACITY15_POP: not a constant expression

//--- capacity15-push.cpp
#include "hal/atomic.h"
namespace atomic = hal::atomic;

constexpr auto fifo = atomic::fifo<15>(hal::gpr<3>());
constexpr auto bad  = fifo.operation().push(hal::gpr<4>());

// CAPACITY15_PUSH: not a constant expression
