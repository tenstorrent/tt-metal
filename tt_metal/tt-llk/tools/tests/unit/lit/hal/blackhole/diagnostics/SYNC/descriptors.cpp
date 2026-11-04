// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/valid.cpp
// RUN: %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DENABLE_LLK_ASSERT %t/valid.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/mutex-acquire.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/post-selector.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/init-initial.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/init-maximum.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/stall-targets.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/stall-conditions.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/semwait-conditions-empty.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/semwait-conditions-range.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/get-list-selector.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/get-operation-selector.cpp 2>&1 | FileCheck %s --check-prefix=INVALID-SELECTOR
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/get-empty.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY
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

// Direct argument-taking encoders remain usable; descriptors cover their runtime codegen.
static_assert(hs::mutex::acquire_operation(hs::Mutex::M4) == 0xa0000004u);
static_assert(hs::mutex::release_operation(hs::Mutex::M3) == 0xa1000003u);
static_assert(hs::semaphore::post_operation(hs::Semaphore::S7) == 0xa4000200u);
static_assert(hs::semaphore::get_operation(hs::Semaphore::S7) == 0xa5000200u);
static_assert(hs::wait::stall_operation(hs::StallTarget::Sfpu, hs::StallCondition::ConfigUnitIdle) == 0xa2801000u);
static_assert(
    hs::wait::semaphore_operation(hs::StallTarget::Math, hs::Semaphore::S1, hs::SemaphoreCondition::WhileZero | hs::SemaphoreCondition::WhileMaximum) ==
    0xa620000bu);

// Template encoders remain usable directly.
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

//--- mutex-acquire.cpp
#include "common.h"

constexpr auto invalid = hs::MutexAcquire {static_cast<hs::Mutex>(1)}.operation();

//--- post-selector.cpp
#include "common.h"

constexpr auto invalid = hs::SemaphorePost {static_cast<hs::Semaphore>(8)}.operation();

//--- init-initial.cpp
#include "common.h"

constexpr auto invalid = hs::SemaphoreInit {hs::Semaphore::S0, 16, 15}.operation();

//--- init-maximum.cpp
#include "common.h"

constexpr auto invalid = hs::SemaphoreInit {hs::Semaphore::S0, 0, 16}.operation();

//--- stall-targets.cpp
#include "common.h"

constexpr auto invalid = hs::StallWait {static_cast<hs::StallTarget>(512), hs::StallCondition::MathIdle}.operation();

//--- stall-conditions.cpp
#include "common.h"

constexpr auto invalid = hs::StallWait {hs::StallTarget::Math, static_cast<hs::StallCondition>(8192)}.operation();

//--- semwait-conditions-empty.cpp
#include "common.h"

constexpr auto invalid = hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S0, static_cast<hs::SemaphoreCondition>(0)}.operation();

//--- semwait-conditions-range.cpp
#include "common.h"

constexpr auto invalid = hs::SemaphoreWait {hs::StallTarget::Math, hs::Semaphore::S0, static_cast<hs::SemaphoreCondition>(4)}.operation();

//--- get-list-selector.cpp
#include "common.h"

constexpr auto invalid = hs::SemaphoreGet {{hs::Semaphore::S1, static_cast<hs::Semaphore>(255)}}.operation();

//--- get-operation-selector.cpp
#include "common.h"

constexpr auto invalid = hs::semaphore::get_operation<hs::Semaphore::S1, static_cast<hs::Semaphore>(8)>();

//--- get-empty.cpp
#include "common.h"

void invalid()
{
    hs::semaphore::get<>();
}
