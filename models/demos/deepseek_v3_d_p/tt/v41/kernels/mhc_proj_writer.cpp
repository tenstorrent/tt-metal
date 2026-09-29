// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the fused V4.1 mHC projection (mhc_proj_compute.cpp). Per tile row r it first streams the weight
// column w [KT tiles, 1 tile] in the reader's block order (BK tiles per block, starting at block r % NB), then
// writes tile row r of the [T, 32] output. Reading the weight here keeps it off the reader's DRAM stream.
//
// compile_time_args = [cb_w, cb_out, KT, BK, TensorAccessorArgs(w), TensorAccessorArgs(out)]
// runtime args      = [w_addr, out_addr, row_start, row_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t row_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t KT = get_compile_time_arg_val(2);
    constexpr uint32_t BK = get_compile_time_arg_val(3);
    constexpr uint32_t NB = KT / BK;
    constexpr auto w_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    if (row_count == 0) {
        return;
    }

    const auto ws = TensorAccessor(w_args, w_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer w(cb_w);
    DataflowBuffer dfb(cb_out);
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        const uint32_t b0 = r % NB;
        for (uint32_t i = 0; i < NB; ++i) {
            const uint32_t k0 = ((b0 + i) % NB) * BK;
            w.reserve_back(BK);
            for (uint32_t b = 0; b < BK; ++b) {
                noc.async_read(ws, w, page, {.page_id = k0 + b}, {.offset_bytes = b * page});
            }
            noc.async_read_barrier();
            w.push_back(BK);
        }
        dfb.wait_front(1);
        noc.async_write(dfb, out, page, {.offset_bytes = 0}, {.page_id = r});
        noc.async_writes_flushed();
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
