// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM kernel for BenchmarkCaseEight (issue #55788).
//
// Both the producer and the consumer side of case eight run this source: the case measures DFB
// *init* time, and the init walk in setup_local_dfb_interfaces is driven entirely by the host
// config blob, not by anything the kernel does. So the body only needs to construct the accessors
// -- there is no data movement to perform and none is wanted, since traffic would add noise to the
// quantity being measured.
//
// Only d0 and d1 are named. Case eight binds 12 DFBs (d0..d11) to each kernel; a hart initializes
// every DFB in its participation mask whether or not the kernel ever references it, so touching
// two is enough to keep the accessor path exercised without inflating the kernel's text.

#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    DataflowBuffer d0(dfb::d0);
    DataflowBuffer d1(dfb::d1);
    (void)d0;
    (void)d1;
}
