// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for readout_compute.cpp. q is `[F, h, Sq, 128]` bf16 tiled, the state's transpose split into bf16 hi / lo
// parts `[F, h, 128, 128]` each; unit u = (f h + head) RT + rt (q rows 32 rt .. of frame f, head). Per unit: q's 4
// column tiles; at the start of each (frame, head) group, that group's 16 hi and 16 lo tiles (compute pops them at
// the group's end). The norm weight rows (4 fp32 tiles, every row the weights) and the ones tile are read once.
//
// Compile-time args: the tensor accessor args of q, hi, lo, w, ones.
// Runtime args: q, hi, lo, w, ones addresses, first unit, number of units, RT.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_q = 0;
constexpr uint32_t cb_hi = 1;
constexpr uint32_t cb_lo = 2;
constexpr uint32_t cb_w = 3;
constexpr uint32_t cb_ones = 4;
constexpr uint32_t HEAD_TILES = 4;
constexpr uint32_t STATE_TILES = 16;
}  // namespace

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t hi_addr = get_arg_val<uint32_t>(1);
    const uint32_t lo_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t ones_addr = get_arg_val<uint32_t>(4);
    const uint32_t first = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);
    const uint32_t rt_count = get_arg_val<uint32_t>(7);

    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto hi_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto lo_args = TensorAccessorArgs<hi_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<lo_args.next_compile_time_args_offset()>();
    constexpr auto ones_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer q_buf(cb_q);
    DataflowBuffer hi_buf(cb_hi);
    DataflowBuffer lo_buf(cb_lo);
    DataflowBuffer w_buf(cb_w);
    DataflowBuffer ones_buf(cb_ones);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto hi = TensorAccessor(hi_args, hi_addr);
    const auto lo = TensorAccessor(lo_args, lo_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto ones = TensorAccessor(ones_args, ones_addr);
    const uint32_t bf16_bytes = q_buf.get_entry_size();
    const uint32_t w_bytes = w_buf.get_entry_size();

    w_buf.reserve_back(HEAD_TILES);
    for (uint32_t c = 0; c < HEAD_TILES; ++c) {
        noc.async_read(w, w_buf, w_bytes, {.page_id = c}, {.offset_bytes = c * w_bytes});
    }
    ones_buf.reserve_back(1);
    noc.async_read(ones, ones_buf, ones_buf.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    w_buf.push_back(HEAD_TILES);
    ones_buf.push_back(1);

    for (uint32_t u = first; u < first + count; ++u) {
        if (u == first || u % rt_count == 0) {
            const uint32_t group = u / rt_count;
            hi_buf.reserve_back(STATE_TILES);
            lo_buf.reserve_back(STATE_TILES);
            for (uint32_t t = 0; t < STATE_TILES; ++t) {
                const uint32_t page = group * STATE_TILES + t;
                noc.async_read(hi, hi_buf, bf16_bytes, {.page_id = page}, {.offset_bytes = t * bf16_bytes});
                noc.async_read(lo, lo_buf, bf16_bytes, {.page_id = page}, {.offset_bytes = t * bf16_bytes});
            }
            noc.async_read_barrier();
            hi_buf.push_back(STATE_TILES);
            lo_buf.push_back(STATE_TILES);
        }
        q_buf.reserve_back(HEAD_TILES);
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc.async_read(q, q_buf, bf16_bytes, {.page_id = u * HEAD_TILES + c}, {.offset_bytes = c * bf16_bytes});
        }
        noc.async_read_barrier();
        q_buf.push_back(HEAD_TILES);
    }
}
