// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A kernel that returns at once, for any processor: what is left of a program's time is its launch (the
// dispatcher's writes per core range and kernel, the go, the firmware's start-up on every core, the done).

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {}
