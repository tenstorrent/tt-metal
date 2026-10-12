// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/dprint.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t block_ct, uint32_t Mt, uint32_t Ct>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_input = TensorAccessor(tensor::grad_input);
    DataflowBuffer output(dfb::output);
    Noc noc;

    const uint32_t tile_bytes = output.get_entry_size();
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = (work / Mt) * block_ct;
        output.wait_front(block_ct);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            noc.async_write(
                output,
                grad_input,
                tile_bytes,
                {.offset_bytes = ct * tile_bytes},
                {.page_id = mt * Ct + ct_start + ct});
        }
        noc.async_write_barrier();
        output.pop_front(block_ct);
        DPRINT("writer: work {} grad_input tiles written\n", work);
    }
}
