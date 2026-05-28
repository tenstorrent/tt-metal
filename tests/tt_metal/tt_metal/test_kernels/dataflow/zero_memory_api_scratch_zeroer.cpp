// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer kernel for the DRAM-with-scratch test (Noc::write_zeros overload 3).
// Zeros a DFB-backed L1 scratch region via overload (1), barriers, then pushes the
// entry so the consumer (DRAM writer) can pick it up.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t scratch_size_bytes = get_arg(args::scratch_size_bytes);

    DataflowBuffer dfb(dfb::scratch);
    dfb.reserve_back(1);

    Noc noc;
    noc.write_zeros(dfb, scratch_size_bytes);
    noc.write_zeros_l1_barrier();

    dfb.push_back(1);
}
