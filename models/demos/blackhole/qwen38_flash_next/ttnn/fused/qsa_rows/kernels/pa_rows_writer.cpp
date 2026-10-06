// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 5 writer: the core's gated tile into page (8 h + t) of the head-major [1, 1, 32, 1536] output
// (the out-projection's activation shard).  CB 16 out (bf16, 1 tile).
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t CB_OUT = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, out_addr);
    {
        FUSED_ZONE("fz_qr_pa_w_tile");
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(page, out, get_read_ptr(CB_OUT));
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
}
