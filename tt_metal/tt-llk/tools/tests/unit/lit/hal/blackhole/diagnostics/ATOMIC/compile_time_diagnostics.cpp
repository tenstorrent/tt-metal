// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-descriptor.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_DESCRIPTOR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/misaligned-masked-write.cpp 2>&1 | FileCheck %s --check-prefix=MISALIGNED_MASKED_WRITE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-counter.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-counter-data.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_COUNTER_DATA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-fifo.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_FIFO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/fifo-increment-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=FIFO_INCREMENT_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-fifo-pop-result.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_FIFO_POP_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-fifo-push-result.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_FIFO_PUSH_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-fifo-wait-result.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_FIFO_WAIT_RESULT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/invalid-lock.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_LOCK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/lock-compare-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=LOCK_COMPARE_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/lock-set-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=LOCK_SET_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-masked-write-address.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_MASKED_WRITE_ADDRESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-masked-write-data.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_MASKED_WRITE_DATA
// clang-format on

//--- invalid-descriptor.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

void invalid_descriptor()
{
    atomic::run<atomic::FetchIncrement {hal::gpr<1>(), hal::gpr<2>(), std::uint8_t {4}, std::uint8_t {32}}>();
}

// INVALID_DESCRIPTOR: error: static assertion failed: invalid atomic descriptor

//--- misaligned-masked-write.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

void misaligned_masked_write()
{
    atomic::store_masked<0xff>(hal::gpr<8>(), hal::gpr<13>());
}

// MISALIGNED_MASKED_WRITE: error: static assertion failed: invalid atomic descriptor

//--- invalid-counter.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Counter<33, 1> counter {hal::gpr<1>()};

void invalid_counter()
{
    atomic::fetch_add<counter>(hal::gpr<2>());
}

// INVALID_COUNTER: error: static assertion failed: invalid atomic counter identity

//--- runtime-counter-data.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Counter<32, 1> counter {hal::gpr<1>()};

void runtime_counter_data(const std::uint32_t index)
{
    atomic::fetch_add<counter>(hal::gpr(index));
}

// RUNTIME_COUNTER_DATA: error: static assertion failed: immediate atomic fetch-add requires hal::gpr<Index>()

//--- invalid-fifo.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Fifo<16, 3> fifo {hal::gpr<3>()};

void invalid_fifo()
{
    atomic::pop_slots<fifo>(hal::gpr<4>());
}

// INVALID_FIFO: error: static assertion failed: invalid atomic FIFO identity

//--- fifo-increment-too-wide.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Fifo<3, 3> fifo {hal::gpr<3>()};

void fifo_increment_too_wide()
{
    atomic::push_slots<fifo, 16>(hal::gpr<4>());
}

// FIFO_INCREMENT_TOO_WIDE: error: static assertion failed: FIFO increment logarithm must be in [0, 15]

//--- runtime-fifo-pop-result.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Fifo<3, 3> fifo {hal::gpr<3>()};

void runtime_fifo_pop_result(const std::uint32_t index)
{
    atomic::pop_slots<fifo>(hal::gpr(index));
}

// RUNTIME_FIFO_POP_RESULT: error: static assertion failed: immediate FIFO pop requires hal::gpr<Index>()

//--- runtime-fifo-push-result.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Fifo<3, 3> fifo {hal::gpr<3>()};

void runtime_fifo_push_result(const std::uint32_t index)
{
    atomic::push_slots<fifo>(hal::gpr(index));
}

// RUNTIME_FIFO_PUSH_RESULT: error: static assertion failed: immediate FIFO push requires hal::gpr<Index>()

//--- runtime-fifo-wait-result.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr atomic::Fifo<3, 3> fifo {hal::gpr<3>()};

void runtime_fifo_wait_result(const std::uint32_t index)
{
    atomic::wait_while<atomic::FifoState::Empty, fifo>(hal::gpr(index));
}

// RUNTIME_FIFO_WAIT_RESULT: error: static assertion failed: immediate FIFO wait requires hal::gpr<Index>()

//--- invalid-lock.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr auto lock = atomic::Lock {hal::gpr<7>(), 4};

void invalid_lock()
{
    atomic::acquire<lock>();
}

// INVALID_LOCK: error: static assertion failed: invalid atomic lock identity

//--- lock-compare-too-wide.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr auto lock = atomic::Lock {hal::gpr<7>()};

void lock_compare_too_wide()
{
    atomic::compare_set<lock, 16, 0>();
}

// LOCK_COMPARE_TOO_WIDE: error: static assertion failed: ATCAS compare value must fit in four bits

//--- lock-set-too-wide.cpp
#include "hal/atomic.h"

namespace atomic = hal::atomic;

constexpr auto lock = atomic::Lock {hal::gpr<7>()};

void lock_set_too_wide()
{
    atomic::compare_set<lock, 0, 16>();
}

// LOCK_SET_TOO_WIDE: error: static assertion failed: ATCAS set value must fit in four bits

//--- runtime-masked-write-address.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

void runtime_masked_write_address(const std::uint32_t index)
{
    atomic::store_masked<0xff>(hal::gpr(index), hal::gpr<12>());
}

// RUNTIME_MASKED_WRITE_ADDRESS: error: static assertion failed: immediate masked write requires hal::gpr<Index>() for its address

//--- runtime-masked-write-data.cpp
#include <cstdint>

#include "hal/atomic.h"

namespace atomic = hal::atomic;

void runtime_masked_write_data(const std::uint32_t index)
{
    atomic::store_masked<0xff>(hal::gpr<8>(), hal::gpr(index));
}

// RUNTIME_MASKED_WRITE_DATA: error: static assertion failed: immediate masked write requires hal::gpr<Index>() for its data quad
