// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/valid.cpp
// RUN: %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT %t/valid.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=1 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=2 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=3 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=4 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=5 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=6 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=7 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=8 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=9 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID-SELECTOR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=10 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY
// clang-format on

// INVALID: error: {{.*}}__builtin_trap(){{.*}} is not a constant expression
// INVALID-SELECTOR: static assertion failed: Semaphore index must be in [0, 7]
// EMPTY: error: no matching function for call to {{.*}}get<>()

//--- common.h
#include "hal/sync.h"

namespace hs = hal::sync;

//--- valid.cpp
#include "common.h"

// Full instruction words, including boundary values and combined masks.
static_assert(hs::MutexAcquire {hs::Mutex::M0}.operation() == 0xa0000000u);
static_assert(hs::MutexAcquire {hs::Mutex::M4}.operation() == 0xa0000004u);
static_assert(hs::MutexRelease {hs::Mutex::M3}.operation() == 0xa1000003u);
static_assert(hs::SemaphoreInit {{hs::Semaphore::S0, hs::Semaphore::S7}, 5, 15}.operation() == 0xa3f50204u);
static_assert(
    hs::SemaphorePost {{hs::Semaphore::S0,
                        hs::Semaphore::S1,
                        hs::Semaphore::S2,
                        hs::Semaphore::S3,
                        hs::Semaphore::S4,
                        hs::Semaphore::S5,
                        hs::Semaphore::S6,
                        hs::Semaphore::S7}}
        .operation() == 0xa40003fcu);
static_assert(hs::SemaphoreGet {hs::Semaphore::S7}.operation() == 0xa5000200u);
static_assert(hs::StallWait {hs::StallTarget::Sfpu, hs::StallCondition::ConfigUnitIdle}.operation() == 0xa2801000u);
static_assert(
    hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S1, hs::SemaphoreCondition::WhileZero | hs::SemaphoreCondition::WhileMaximum}.operation() ==
    0xa620000bu);

// Existing accessors and execution forms remain usable.
static_assert(hs::mutex::acquire_operation<hs::Mutex::M4>() == 0xa0000004u);
static_assert(
    hs::semaphore::post_operation<
        hs::Semaphore::S0,
        hs::Semaphore::S1,
        hs::Semaphore::S2,
        hs::Semaphore::S3,
        hs::Semaphore::S4,
        hs::Semaphore::S5,
        hs::Semaphore::S6,
        hs::Semaphore::S7>() == 0xa40003fcu);

// Selectors become bits internally; selecting the same semaphore twice is idempotent.
static_assert(hs::semaphore::get_operation<hs::Semaphore::S0>() == 0xa5000004u);
static_assert(hs::semaphore::get_operation<hs::Semaphore::S1>() == 0xa5000008u);
static_assert(hs::semaphore::get_operation<hs::Semaphore::S1, hs::Semaphore::S3>() == 0xa5000028u);
static_assert(hs::semaphore::get_operation<hs::Semaphore::S1, hs::Semaphore::S1>() == 0xa5000008u);
static_assert(hs::SemaphoreGet {{hs::Semaphore::S1, hs::Semaphore::S1}}.operation() == 0xa5000008u);
static_assert(hs::semaphore::init_operation<5, 15, hs::Semaphore::S0, hs::Semaphore::S7>() == 0xa3f50204u);
static_assert(hs::wait::semaphore_operation<hs::StallTarget::Math, hs::SemaphoreCondition::WhileZero, hs::Semaphore::S1, hs::Semaphore::S3>() == 0xa6200029u);

//--- invalid.cpp
#include "common.h"

#if INVALID_CASE == 1
constexpr auto invalid = hs::MutexAcquire {static_cast<hs::Mutex>(1)}.operation();
#elif INVALID_CASE == 2
constexpr auto invalid = hs::SemaphorePost {static_cast<hs::Semaphore>(8)}.operation();
#elif INVALID_CASE == 3
constexpr auto invalid = hs::SemaphoreInit {hs::Semaphore::S0, 16, 15}.operation();
#elif INVALID_CASE == 4
constexpr auto invalid = hs::SemaphoreInit {hs::Semaphore::S0, 0, 16}.operation();
#elif INVALID_CASE == 5
constexpr auto invalid = hs::StallWait {static_cast<hs::StallTarget>(512), hs::StallCondition::MathIdle}.operation();
#elif INVALID_CASE == 6
constexpr auto invalid = hs::StallWait {hs::StallTarget::Math, static_cast<hs::StallCondition>(8192)}.operation();
#elif INVALID_CASE == 7
constexpr auto invalid = hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S0, static_cast<hs::SemaphoreCondition>(0)}.operation();
#elif INVALID_CASE == 8
constexpr auto invalid = hs::SemaphoreGet {{hs::Semaphore::S1, static_cast<hs::Semaphore>(255)}}.operation();
#elif INVALID_CASE == 9
constexpr auto invalid = hs::semaphore::get_operation<hs::Semaphore::S1, static_cast<hs::Semaphore>(8)>();
#elif INVALID_CASE == 10
void invalid()
{
    hs::semaphore::get<>();
}
#endif
