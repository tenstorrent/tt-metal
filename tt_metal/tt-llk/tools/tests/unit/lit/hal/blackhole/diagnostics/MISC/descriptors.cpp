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
// clang-format on

// INVALID: error: {{.*}}__builtin_trap(){{.*}} is not a constant expression

//--- common.h
#include "hal/misc.h"

namespace hm = hal::misc;

//--- valid.cpp
#include "common.h"

// Full instruction words, including boundary values and combined masks.
static_assert(hm::FlushTdma {}.operation() == 0x46000000u);
static_assert(hm::FlushTdma {hm::FlushScope::Packer | hm::FlushScope::Unpacker0}.operation() == 0x4600000au);
static_assert(hm::ResetTdma {}.operation() == 0x44000000u);
static_assert(hm::TbufCommand {}.operation() == 0x4b000000u);
static_assert(hm::ResourceDeclaration {15, 511, 2047}.operation() == 0x05ffffffu);

// Existing accessors and execution forms remain usable.
static_assert(hm::ResourceDeclaration {15, 511, 2047}.get_operation() == 0x05ffffffu);
static_assert(hm::flush_tdma_operation<hm::FlushScope::All>() == 0x46000000u);

//--- invalid.cpp
#include "common.h"

#if INVALID_CASE == 1
constexpr auto invalid = hm::FlushTdma {static_cast<hm::FlushScope>(16)}.operation();
#elif INVALID_CASE == 2
constexpr auto invalid = hm::ResourceDeclaration {16, 0, 1}.operation();
#elif INVALID_CASE == 3
constexpr auto invalid = hm::ResourceDeclaration {0, 512, 1}.operation();
#elif INVALID_CASE == 4
constexpr auto invalid = hm::ResourceDeclaration {0, 0, 2048}.operation();
#endif
