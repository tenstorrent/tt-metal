// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %s 2>&1 | FileCheck %s
// CHECK: error: static assertion failed: expected an AddrCounter or MathCounter descriptor

#include "hal/tensix_counters.h"

void probe()
{
    hal::counters::set<5>();
}
