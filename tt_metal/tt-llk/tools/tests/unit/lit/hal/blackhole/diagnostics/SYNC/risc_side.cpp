// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// DEFINE: %{diagnose} = %{blackhole_tensix_diagnose} %{blackhole_unpack_thread}
// RUN: %split-file %s %t
// RUN: not %{diagnose} %t/mailbox-write.cpp 2>&1 | FileCheck %s --check-prefix=MAILBOX
// RUN: not %{diagnose} %t/mailbox-read.cpp 2>&1 | FileCheck %s --check-prefix=MAILBOX
// RUN: not %{diagnose} %t/mailbox-has-data.cpp 2>&1 | FileCheck %s --check-prefix=MAILBOX
// RUN: not %{diagnose} %t/gpr-window-index.cpp 2>&1 | FileCheck %s --check-prefix=GPR
// RUN: not %{diagnose} %t/gpr-tensix-wait.cpp 2>&1 | FileCheck %s --check-prefix=TENSIX_WAIT
// RUN: not %{diagnose} %t/gpr-tensix-read.cpp 2>&1 | FileCheck %s --check-prefix=TENSIX_READ

// MAILBOX: error: static assertion failed: mailbox RISC must be Brisc, Unpack, Math, or Pack
// GPR: error: static assertion failed: Scalar-unit GPR index must be in [0, 63]
// TENSIX_WAIT: error: static assertion failed: completion applies to MMIO GPR stores; Tensix writes are ordered
// TENSIX_READ: error: static assertion failed: value-returning GPR reads require Access::MMIO
// clang-format on

//--- mailbox-write.cpp
#include "hal/sync.h"

void test()
{
    hal::sync::mailbox::write<static_cast<hal::sync::Riscv>(4)>(1);
}

//--- mailbox-read.cpp
#include "hal/sync.h"

void test()
{
    static_cast<void>(hal::sync::mailbox::read<static_cast<hal::sync::Riscv>(4)>());
}

//--- mailbox-has-data.cpp
#include "hal/sync.h"

void test()
{
    static_cast<void>(hal::sync::mailbox::has_data<static_cast<hal::sync::Riscv>(4)>());
}

//--- gpr-window-index.cpp
#include "hal/scalar.h"

void test()
{
    hal::scalar::set<hal::scalar::Access::MMIO>(hal::gpr<64>(), 1);
}

//--- gpr-tensix-wait.cpp
#include "hal/scalar.h"

void test()
{
    hal::scalar::set<hal::scalar::Access::Tensix, hal::scalar::Completion::Wait>(hal::gpr<4>(), 1);
}

//--- gpr-tensix-read.cpp
#include "hal/scalar.h"

void test()
{
    static_cast<void>(hal::scalar::read<hal::scalar::Access::Tensix>(hal::gpr<4>()));
}
