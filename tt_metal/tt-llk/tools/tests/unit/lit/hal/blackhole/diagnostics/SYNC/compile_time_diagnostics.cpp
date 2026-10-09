// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mutex-acquire.cpp 2>&1 | FileCheck %s --check-prefix=MUTEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mutex-release-operation.cpp 2>&1 | FileCheck %s --check-prefix=MUTEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/init-initial.cpp 2>&1 | FileCheck %s --check-prefix=INIT_INITIAL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/init-maximum.cpp 2>&1 | FileCheck %s --check-prefix=INIT_MAXIMUM
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/post-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/get-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/read-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/post-list-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/post-access.cpp 2>&1 | FileCheck %s --check-prefix=SEM_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/get-runtime-access.cpp 2>&1 | FileCheck %s --check-prefix=SEM_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stall-targets.cpp 2>&1 | FileCheck %s --check-prefix=STALL_TARGETS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stall-conditions.cpp 2>&1 | FileCheck %s --check-prefix=STALL_CONDITIONS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/semwait-targets.cpp 2>&1 | FileCheck %s --check-prefix=SEMWAIT_TARGETS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/semwait-conditions.cpp 2>&1 | FileCheck %s --check-prefix=SEMWAIT_CONDITIONS
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// MUTEX: error: static assertion failed: Blackhole mutex index must be 0 or in [2, 4]
// INIT_INITIAL: error: static assertion failed: SEMINIT initial value must fit in four bits
// INIT_MAXIMUM: error: static assertion failed: SEMINIT maximum value must fit in four bits
// SEM_INDEX: error: static assertion failed: Semaphore index must be in [0, 7]
// SEM_ACCESS: error: static assertion failed: Semaphore access must be MMIO or Tensix
// STALL_TARGETS: error: static assertion failed: STALLWAIT target mask must fit in nine bits
// STALL_CONDITIONS: error: static assertion failed: Blackhole STALLWAIT condition mask must fit in 13 bits
// SEMWAIT_TARGETS: error: static assertion failed: SEMWAIT target mask must fit in nine bits
// SEMWAIT_CONDITIONS: error: static assertion failed: SEMWAIT requires WhileZero, WhileMaximum, or both

//--- common.h
#pragma once

#include <cstdint>

#include "hal/sync.h"

namespace sync = hal::sync;

//--- mutex-acquire.cpp
#include "common.h"

void probe()
{
    sync::mutex::acquire<static_cast<sync::Mutex>(1)>();
}

//--- mutex-release-operation.cpp
#include "common.h"

constexpr std::uint32_t operation = sync::mutex::release_operation<static_cast<sync::Mutex>(5)>();

//--- init-initial.cpp
#include "common.h"

void probe()
{
    sync::semaphore::init<16, 15, sync::Semaphore::S1>();
}

//--- init-maximum.cpp
#include "common.h"

constexpr std::uint32_t operation = sync::semaphore::init_operation<0, 16, sync::Semaphore::S1>();

//--- post-index.cpp
#include "common.h"

void probe()
{
    sync::semaphore::post<sync::Access::Tensix, static_cast<sync::Semaphore>(8)>();
}

//--- get-index.cpp
#include "common.h"

void probe()
{
    sync::semaphore::get<sync::Access::MMIO, static_cast<sync::Semaphore>(8)>();
}

//--- read-index.cpp
#include "common.h"

std::uint8_t probe()
{
    return sync::semaphore::read<static_cast<sync::Semaphore>(8)>();
}

//--- post-list-index.cpp
#include "common.h"

void probe()
{
    sync::semaphore::post<sync::Semaphore::S1, static_cast<sync::Semaphore>(8)>();
}

//--- post-access.cpp
#include "common.h"

void probe()
{
    sync::semaphore::post<static_cast<sync::Access>(2), sync::Semaphore::S0>();
}

//--- get-runtime-access.cpp
#include "common.h"

void probe(const sync::Semaphore semaphore)
{
    sync::semaphore::get<static_cast<sync::Access>(2)>(semaphore);
}

//--- stall-targets.cpp
#include "common.h"

void probe()
{
    sync::wait::stall<static_cast<sync::StallTarget>(0x200), sync::StallCondition::MathIdle>();
}

//--- stall-conditions.cpp
#include "common.h"

void probe()
{
    sync::wait::stall<sync::StallTarget::Math, static_cast<sync::StallCondition>(0x2000)>();
}

//--- semwait-targets.cpp
#include "common.h"

void probe()
{
    sync::wait::semaphore<static_cast<sync::StallTarget>(0x200), sync::SemaphoreCondition::WhileZero, sync::Semaphore::S1>();
}

//--- semwait-conditions.cpp
#include "common.h"

void probe()
{
    sync::wait::semaphore<sync::StallTarget::Math, static_cast<sync::SemaphoreCondition>(0), sync::Semaphore::S1>();
}
