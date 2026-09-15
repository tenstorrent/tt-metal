// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aa = TensorAccessorArgs<0>();
    constexpr auto ba = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    auto a = TensorAccessor(aa, get_arg_val<uint32_t>(0));
    auto b = TensorAccessor(ba, get_arg_val<uint32_t>(1));
    Noc noc;
    DataflowBuffer acb(0), bcb(1);
    acb.reserve_back(4);
    bcb.reserve_back(16);
    for (uint32_t i = 0; i < 4; ++i) {
        noc.async_read(a, acb, 2048, {.page_id = i}, {.offset_bytes = i * 2048});
    }
    for (uint32_t i = 0; i < 16; ++i) {
        noc.async_read(b, bcb, 2048, {.page_id = i}, {.offset_bytes = i * 2048});
    }
    cb_reserve_back(2, 4);
    auto* scores = reinterpret_cast<volatile uint32_t*>(get_write_ptr(2));
    for (uint32_t i = 0; i < 4096; ++i) {
        scores[i] = 0xc1000000;  // -8.0f, fixed negative scores for scheduling probe.
    }
    cb_push_back(2, 4);
    noc.async_read_barrier();
    acb.push_back(4);
    bcb.push_back(16);
}
