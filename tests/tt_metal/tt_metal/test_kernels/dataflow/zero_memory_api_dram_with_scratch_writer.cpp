// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Consumer kernel for the DRAM-with-scratch test (Noc::write_zeros overload 3).
// Waits for the producer to publish a pre-zeroed entry on dfb::scratch, then loops
// noc.write_zeros over [page_start, page_end) passing the scratch as the working
// buffer. Flushes with the matching DRAM barrier, then pops the scratch entry.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t page_start = get_arg(args::page_start);
    const uint32_t page_end = get_arg(args::page_end);
    const uint32_t page_size = get_arg(args::page_size);

    DataflowBuffer dfb(dfb::scratch);
    dfb.wait_front(1);

    const auto out = TensorAccessor(ta::out);
    Noc noc;

    for (uint32_t p = page_start; p < page_end; ++p) {
        noc.write_zeros(out, page_size, {.page_id = p}, dfb);
    }
    noc.write_zeros_dram_barrier();

    dfb.pop_front(1);
}
