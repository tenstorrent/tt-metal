// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for the DFB entry-distribution probe (issue #55788).
//
// Binds N DFBs through the *standard* Metal 2.0 binding API (ProducerOf), rather than the
// experimental explicit producer_risc_mask the worst-case benchmarks use. The point of the probe
// is what the host does with those bindings: a DFB bound to a kernel inherits that kernel's whole
// risc mask (program_spec.cpp:2640), so every thread of this kernel becomes a producer of every
// DFB bound here.
//
// The body only constructs each DFB — no data movement. Entry distribution is decided by the host
// config, so the walk in setup_local_dfb_interfaces happens regardless of what runs here.

#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer d0(dfb::d0);
    DataflowBuffer d1(dfb::d1);
    (void)d0;
    (void)d1;
}
