// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mutex-acquire.cpp 2>&1 | FileCheck %s --check-prefix=MUTEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mutex-release-operation.cpp 2>&1 | FileCheck %s --check-prefix=MUTEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/init-empty-mask.cpp 2>&1 | FileCheck %s --check-prefix=INIT_MASK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/init-initial.cpp 2>&1 | FileCheck %s --check-prefix=INIT_INITIAL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/init-maximum.cpp 2>&1 | FileCheck %s --check-prefix=INIT_MAXIMUM
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/post-empty-mask.cpp 2>&1 | FileCheck %s --check-prefix=POST_MASK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/get-empty-mask.cpp 2>&1 | FileCheck %s --check-prefix=GET_MASK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/post-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/get-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/read-index.cpp 2>&1 | FileCheck %s --check-prefix=SEM_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/post-access.cpp 2>&1 | FileCheck %s --check-prefix=SEM_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/get-runtime-access.cpp 2>&1 | FileCheck %s --check-prefix=SEM_ACCESS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stall-targets.cpp 2>&1 | FileCheck %s --check-prefix=STALL_TARGETS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stall-conditions.cpp 2>&1 | FileCheck %s --check-prefix=STALL_CONDITIONS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/semwait-targets.cpp 2>&1 | FileCheck %s --check-prefix=SEMWAIT_TARGETS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/semwait-empty-mask.cpp 2>&1 | FileCheck %s --check-prefix=SEMWAIT_MASK
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/semwait-conditions.cpp 2>&1 | FileCheck %s --check-prefix=SEMWAIT_CONDITIONS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stream-slot.cpp 2>&1 | FileCheck %s --check-prefix=STREAM_SLOT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stream-id-slot.cpp 2>&1 | FileCheck %s --check-prefix=STREAM_SLOT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stream-group.cpp 2>&1 | FileCheck %s --check-prefix=STREAM_GROUP
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/stream-number.cpp 2>&1 | FileCheck %s --check-prefix=STREAM_NUMBER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/phase-target-width.cpp 2>&1 | FileCheck %s --check-prefix=TARGET_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/messages-target-width.cpp 2>&1 | FileCheck %s --check-prefix=TARGET_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/combined-target-width.cpp 2>&1 | FileCheck %s --check-prefix=TARGET_WIDTH
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/target-select.cpp 2>&1 | FileCheck %s --check-prefix=TARGET_SELECT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/streamwait-low-target.cpp 2>&1 | FileCheck %s --check-prefix=LOW_TARGET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/streamwait-targets.cpp 2>&1 | FileCheck %s --check-prefix=STREAMWAIT_TARGETS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/streamwait-slot.cpp 2>&1 | FileCheck %s --check-prefix=STREAM_SLOT
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// MUTEX: error: static assertion failed: Blackhole mutex index must be 0 or in [2, 4]
// INIT_MASK: error: static assertion failed: SEMINIT requires at least one semaphore
// INIT_INITIAL: error: static assertion failed: SEMINIT initial value must fit in four bits
// INIT_MAXIMUM: error: static assertion failed: SEMINIT maximum value must fit in four bits
// POST_MASK: error: static assertion failed: SEMPOST requires at least one semaphore
// GET_MASK: error: static assertion failed: SEMGET requires at least one semaphore
// SEM_INDEX: error: static assertion failed: Semaphore index must be in [0, 7]
// SEM_ACCESS: error: static assertion failed: Semaphore access must be MMIO or Tensix
// STALL_TARGETS: error: static assertion failed: STALLWAIT target mask must fit in nine bits
// STALL_CONDITIONS: error: static assertion failed: Blackhole STALLWAIT condition mask must fit in 13 bits
// SEMWAIT_TARGETS: error: static assertion failed: SEMWAIT target mask must fit in nine bits
// SEMWAIT_MASK: error: static assertion failed: SEMWAIT requires at least one semaphore
// SEMWAIT_CONDITIONS: error: static assertion failed: SEMWAIT requires WhileZero, WhileMaximum, or both
// STREAM_SLOT: error: static assertion failed: STREAMWAIT slot must be in [0, 3]
// STREAM_GROUP: error: static assertion failed: NoC stream group must fit in three bits
// STREAM_NUMBER: error: static assertion failed: NoC stream number must fit in three bits
// TARGET_WIDTH: error: static assertion failed: STREAMWAIT target exceeds the selected counter width
// TARGET_SELECT: error: static assertion failed: STREAMWAIT target must be Phase or MessagesReceived
// LOW_TARGET: error: static assertion failed: STREAMWAIT low target must fit in ten bits
// STREAMWAIT_TARGETS: error: static assertion failed: STREAMWAIT target mask must fit in nine bits

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

//--- init-empty-mask.cpp
#include "common.h"

void probe()
{
    sync::semaphore::init<sync::SemaphoreMask::None, 0, 1>();
}

//--- init-initial.cpp
#include "common.h"

void probe()
{
    sync::semaphore::init<sync::SemaphoreMask::MathPack, 16, 15>();
}

//--- init-maximum.cpp
#include "common.h"

constexpr std::uint32_t operation = sync::semaphore::init_operation<sync::SemaphoreMask::MathPack, 0, 16>();

//--- post-empty-mask.cpp
#include "common.h"

void probe()
{
    sync::semaphore::post<sync::SemaphoreMask::None>();
}

//--- get-empty-mask.cpp
#include "common.h"

void probe()
{
    sync::semaphore::get<sync::SemaphoreMask::None>();
}

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
    sync::wait::semaphore<static_cast<sync::StallTarget>(0x200), sync::SemaphoreMask::MathPack, sync::SemaphoreCondition::WhileZero>();
}

//--- semwait-empty-mask.cpp
#include "common.h"

void probe()
{
    sync::wait::semaphore<sync::StallTarget::Math, sync::SemaphoreMask::None, sync::SemaphoreCondition::WhileZero>();
}

//--- semwait-conditions.cpp
#include "common.h"

void probe()
{
    sync::wait::semaphore<sync::StallTarget::Math, sync::SemaphoreMask::MathPack, static_cast<sync::SemaphoreCondition>(0)>();
}

//--- stream-slot.cpp
#include "common.h"

void probe()
{
    sync::wait::configure_stream<static_cast<sync::StreamSlot>(4), 0, 0>();
}

//--- stream-id-slot.cpp
#include "common.h"

void probe(const sync::StreamId stream_id)
{
    sync::wait::configure_stream<static_cast<sync::StreamSlot>(4)>(stream_id);
}

//--- stream-group.cpp
#include "common.h"

void probe()
{
    sync::wait::configure_stream<sync::StreamSlot::S0, 8, 0>();
}

//--- stream-number.cpp
#include "common.h"

void probe()
{
    sync::wait::configure_stream<sync::StreamSlot::S0, 0, 8>();
}

//--- phase-target-width.cpp
#include "common.h"

void probe()
{
    sync::wait::configure_stream_target<sync::StreamTarget::Phase, 1u << 20>();
}

//--- messages-target-width.cpp
#include "common.h"

void probe()
{
    sync::wait::configure_stream_target<sync::StreamTarget::MessagesReceived, 1u << 17>();
}

//--- combined-target-width.cpp
#include "common.h"

void probe(const sync::StreamId stream_id)
{
    sync::wait::configure_and_wait_stream<sync::StallTarget::Math, sync::StreamSlot::S0, sync::StreamTarget::MessagesReceived, 1u << 17>(stream_id);
}

//--- target-select.cpp
#include "common.h"

void probe()
{
    sync::wait::configure_stream_target<static_cast<sync::StreamTarget>(2), 0>();
}

//--- streamwait-low-target.cpp
#include "common.h"

void probe()
{
    sync::wait::stream<sync::StallTarget::Math, sync::StreamSlot::S0, sync::StreamTarget::Phase, 1024>();
}

//--- streamwait-targets.cpp
#include "common.h"

constexpr std::uint32_t operation = sync::wait::stream_operation<static_cast<sync::StallTarget>(0x200), sync::StreamSlot::S0, sync::StreamTarget::Phase, 0>();

//--- streamwait-slot.cpp
#include "common.h"

void probe()
{
    sync::wait::stream<sync::StallTarget::Math, static_cast<sync::StreamSlot>(4), sync::StreamTarget::Phase, 0>();
}
