// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Consumer half of the unified Noc::write_zeros end-to-end test.
// Pairs with zero_memory_api_l1_producer.cpp: the producer tests overload (1) on a
// DFB-resident L1 region (CPU-stamp + write_zeros + verify), and as a side effect
// leaves that DFB entry filled with zeros. This consumer wait_fronts on the same
// DFB and consumes the now-zero entry as the pre-zeroed scratch for overload (2),
// looping over the DRAM page range. The single shared DFB scratch eliminates the
// need for a separate scratch_zeroer kernel — the L1 zero IS the scratch fill.

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
