// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 5 writer: the core's gated tile into page (8 h + t) of the head-major [1, 1, 32, 1536] output
// (the out-projection's activation shard).  CB 16 out (bf16, 1 tile).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;
    {
        CircularBuffer(cb_out).wait_front(1);
        noc.async_write(CircularBuffer(cb_out), out, out.get_aligned_page_size(), {}, {.page_id = page});
        noc.async_write_barrier();
        CircularBuffer(cb_out).pop_front(1);
    }
}
