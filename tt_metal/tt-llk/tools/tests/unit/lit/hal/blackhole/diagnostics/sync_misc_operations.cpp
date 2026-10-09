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
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=9 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=10 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=11 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=12 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=13 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} -DINVALID_CASE=14 %t/invalid.cpp 2>&1 | FileCheck %s --check-prefix=INVALID
// clang-format on

// INVALID: error: {{.*}}__builtin_trap(){{.*}} is not a constant expression

//--- common.h
#include "hal/misc.h"
#include "hal/sync.h"

namespace hs = hal::sync;
namespace hm = hal::misc;

//--- valid.cpp
#include "common.h"

// Full instruction words, including boundary values and combined masks.
static_assert(hs::MutexAcquire {hs::Mutex::M0}.operation() == 0xa0000000u);
static_assert(hs::MutexAcquire {hs::Mutex::M4}.operation() == 0xa0000004u);
static_assert(hs::MutexRelease {hs::Mutex::M3}.operation() == 0xa1000003u);
static_assert(hs::SemaphoreInit {hs::SemaphoreMask::S0 | hs::SemaphoreMask::S7, 5, 15}.operation() == 0xa3f50204u);
static_assert(hs::SemaphorePost {hs::SemaphoreMask::All}.operation() == 0xa40003fcu);
static_assert(hs::SemaphoreGet {hs::SemaphoreMask::S7}.operation() == 0xa5000200u);
static_assert(hs::StallWait {hs::StallTarget::Sfpu, hs::StallCondition::ConfigUnitIdle}.operation() == 0xa2801000u);
static_assert(
    hs::SemaphoreWait {hs::StallTarget::Math, hs::SemaphoreMask::S1, hs::SemaphoreCondition::WhileZero | hs::SemaphoreCondition::WhileMaximum}.operation() ==
    0xa620000bu);
static_assert(hs::StreamWait {hs::StallTarget::All, hs::StreamSlot::S3, hs::StreamTarget::MessagesReceived, 1023}.operation() == 0xa7ffbffbu);
static_assert(hm::FlushTdma {}.operation() == 0x46000000u);
static_assert(hm::FlushTdma {hm::FlushScope::Packer | hm::FlushScope::Unpacker0}.operation() == 0x4600000au);
static_assert(hm::ResetTdma {}.operation() == 0x44000000u);
static_assert(hm::TbufCommand {}.operation() == 0x4b000000u);
static_assert(hm::ResourceDeclaration {15, 511, 2047}.operation() == 0x05ffffffu);

// Existing accessors and execution forms remain usable.
static_assert(hm::ResourceDeclaration {15, 511, 2047}.get_operation() == 0x05ffffffu);
static_assert(hs::mutex::acquire_operation<hs::Mutex::M4>() == 0xa0000004u);
static_assert(hs::semaphore::post_operation<hs::SemaphoreMask::All>() == 0xa40003fcu);
static_assert(hm::flush_tdma_operation<hm::FlushScope::All>() == 0x46000000u);

//--- invalid.cpp
#include "common.h"

#if INVALID_CASE == 1
constexpr auto invalid = hs::MutexAcquire {static_cast<hs::Mutex>(1)}.operation();
#elif INVALID_CASE == 2
constexpr auto invalid = hs::SemaphorePost {hs::SemaphoreMask::None}.operation();
#elif INVALID_CASE == 3
constexpr auto invalid = hs::SemaphoreInit {hs::SemaphoreMask::S0, 16, 15}.operation();
#elif INVALID_CASE == 4
constexpr auto invalid = hs::SemaphoreInit {hs::SemaphoreMask::S0, 0, 16}.operation();
#elif INVALID_CASE == 5
constexpr auto invalid = hs::StallWait {static_cast<hs::StallTarget>(512), hs::StallCondition::MathIdle}.operation();
#elif INVALID_CASE == 6
constexpr auto invalid = hs::StallWait {hs::StallTarget::Math, static_cast<hs::StallCondition>(8192)}.operation();
#elif INVALID_CASE == 7
constexpr auto invalid = hs::SemaphoreWait {hs::StallTarget::Math, hs::SemaphoreMask::S0, static_cast<hs::SemaphoreCondition>(0)}.operation();
#elif INVALID_CASE == 8
constexpr auto invalid = hs::StreamWait {hs::StallTarget::Unpack, static_cast<hs::StreamSlot>(4), hs::StreamTarget::Phase, 0}.operation();
#elif INVALID_CASE == 9
constexpr auto invalid = hs::StreamWait {hs::StallTarget::Unpack, hs::StreamSlot::S0, static_cast<hs::StreamTarget>(2), 0}.operation();
#elif INVALID_CASE == 10
constexpr auto invalid = hs::StreamWait {hs::StallTarget::Unpack, hs::StreamSlot::S0, hs::StreamTarget::Phase, 1024}.operation();
#elif INVALID_CASE == 11
constexpr auto invalid = hm::FlushTdma {static_cast<hm::FlushScope>(16)}.operation();
#elif INVALID_CASE == 12
constexpr auto invalid = hm::ResourceDeclaration {16, 0, 1}.operation();
#elif INVALID_CASE == 13
constexpr auto invalid = hm::ResourceDeclaration {0, 512, 1}.operation();
#elif INVALID_CASE == 14
constexpr auto invalid = hm::ResourceDeclaration {0, 0, 2048}.operation();
#endif
