// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads two interleaved input tensors (gate -> c_0, up -> c_1) a tile at a time,
// for the standalone situ_glu SFPU op test.
//   runtime_args    = [gate_addr, up_addr, num_pages, start_id]
//   compile_time_args = gate TensorAccessorArgs ++ up TensorAccessorArgs

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t gate_addr = get_arg_val<uint32_t>(0);
    const uint32_t up_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr auto gate_args = TensorAccessorArgs<0>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_gate = 0;
    constexpr uint32_t cb_up = 1;

    const uint32_t gate_page = get_local_cb_interface(cb_gate).fifo_page_size;
    const uint32_t up_page = get_local_cb_interface(cb_up).fifo_page_size;

    const auto sg = TensorAccessor(gate_args, gate_addr);
    const auto su = TensorAccessor(up_args, up_addr);

    Noc noc;
    DataflowBuffer dfb_gate(cb_gate);
    DataflowBuffer dfb_up(cb_up);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_gate.reserve_back(1);
        dfb_up.reserve_back(1);
        noc.async_read(sg, dfb_gate, gate_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(su, dfb_up, up_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_gate.push_back(1);
        dfb_up.push_back(1);
    }
}
