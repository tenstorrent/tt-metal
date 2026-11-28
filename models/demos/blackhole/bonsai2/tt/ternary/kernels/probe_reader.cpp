// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe reader: one bf16 activation tile into c_0, `num_w` ternary weight pages (256 B each) into c_1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_w = get_arg_val<uint32_t>(2);
    constexpr auto act_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    const auto act = TensorAccessor(act_args, act_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    Noc noc;
    DataflowBuffer dfb0(tt::CBIndex::c_0);
    DataflowBuffer dfb1(tt::CBIndex::c_1);
    dfb0.reserve_back(1);
    noc.async_read(
        act, dfb0, get_local_cb_interface(tt::CBIndex::c_0).fifo_page_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb0.push_back(1);
    const uint32_t w_bytes = get_local_cb_interface(tt::CBIndex::c_1).fifo_page_size;
    for (uint32_t i = 0; i < num_w; ++i) {
        dfb1.reserve_back(1);
        noc.async_read(w, dfb1, w_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb1.push_back(1);
    }
}
