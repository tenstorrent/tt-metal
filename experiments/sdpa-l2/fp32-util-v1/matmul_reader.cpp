// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t kt = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer acb(0), bcb(1);
    const uint32_t a_page = get_local_cb_interface(0).fifo_page_size;
    acb.reserve_back(kt);
    bcb.reserve_back(kt * 4);
    for (uint32_t i = 0; i < kt; ++i) {
        noc.async_read(a, acb, a_page, {.page_id = i}, {.offset_bytes = i * a_page});
    }
    for (uint32_t i = 0; i < kt * 4; ++i) {
        noc.async_read(b, bcb, 2048, {.page_id = i}, {.offset_bytes = i * 2048});
    }
    noc.async_read_barrier();
    acb.push_back(kt);
    bcb.push_back(kt * 4);
}
