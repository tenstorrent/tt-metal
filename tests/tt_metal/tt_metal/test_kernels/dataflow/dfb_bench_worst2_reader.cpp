// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for the worst-case-two DFB init benchmark.
//
// Each of the four reader kernels (A/B/C/D) is a single-thread DM that acts
// as STRIDED (1 of 1) producer on 8 DFBs using the 1-to-1 ALL consumer pattern.
// All four reader kernel instances share this source file; the specific DFBs
// bound to each instance differ only in their host-side KernelSpec bindings.
//
// All DFBs are finished immediately without posting entries.

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer dfb0(dfb::out0);
    DataflowBuffer dfb1(dfb::out1);
    DataflowBuffer dfb2(dfb::out2);
    DataflowBuffer dfb3(dfb::out3);
    DataflowBuffer dfb4(dfb::out4);
    DataflowBuffer dfb5(dfb::out5);
    DataflowBuffer dfb6(dfb::out6);
    DataflowBuffer dfb7(dfb::out7);
    dfb0.finish();
    dfb1.finish();
    dfb2.finish();
    dfb3.finish();
    dfb4.finish();
    dfb5.finish();
    dfb6.finish();
    dfb7.finish();
}
