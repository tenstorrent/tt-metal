// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 DFB consumer shared by the device-zero tests.
//
//   ZERO_DRAM defined   -> consume the pre-zeroed L1 scratch entry and stream zeros to a DRAM
//                          tensor's pages via overload (2) (test_zero_memory_api.cpp).
//   ZERO_DRAM undefined -> a bare consumer that just drains the entry the producer pushes.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef ZERO_DRAM
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#endif
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef ZERO_DRAM
    const uint32_t page_start = get_arg(args::page_start);
    const uint32_t page_end = get_arg(args::page_end);
    const uint32_t page_size = get_arg(args::page_size);
#endif

    DataflowBuffer dfb(dfb::scratch);
    dfb.wait_front(1);

#ifdef ZERO_DRAM
    const auto out = TensorAccessor(tensor::out);
    Noc noc;
    for (uint32_t p = page_start; p < page_end; ++p) {
        noc.async_write_zeros(out, page_size, {.page_id = p}, dfb);
    }
    noc.write_zeros_dram_barrier();
#endif

    dfb.pop_front(1);
}
