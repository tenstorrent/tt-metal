// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe writer: `num_out` bf16 tiles from c_16 to the output tensor.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_out = get_arg_val<uint32_t>(1);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer dfb(tt::CBIndex::c_16);
    const uint32_t bytes = get_local_cb_interface(tt::CBIndex::c_16).fifo_page_size;
    for (uint32_t i = 0; i < num_out; ++i) {
        dfb.wait_front(1);
        noc.async_write(dfb, out, bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb.pop_front(1);
    }
}
