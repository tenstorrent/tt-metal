// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for conv_temporal_compute.cpp. y is `[F, H, S, 128]` fp32 tiled; unit u = (f H + head) RT + rt. Per unit and
// column tile c: y[f + dt - 2] for the dt in 0..4 whose frame exists (TEMPORAL), else y[f]. The temporal weights
// `[5, 1, 32, C]` (each tile row the channel weights) and the ones tile are read once.
//
// ROWS (no temporal taps): y is instead the token-major rows `[1, 1, R, heads 128]` (bf16 or fp32), frame f's rows
// from tile row `row_tile0 + f stride_tiles`, whole tiles, so unit u's tiles are read in place (no head-layout copy).
//
// Compile-time args: temporal, l2, channel tiles, rows, then the tensor accessor args of y, w, ones.
// Runtime args: y, w, ones addresses, first unit, number of units, frames, heads, RT, row_tile0, stride_tiles.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_y = 0;
constexpr uint32_t cb_w = 1;
constexpr uint32_t cb_ones = 2;
constexpr uint32_t HEAD_TILES = 4;
constexpr uint32_t TAPS = 5;
}  // namespace

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t ones_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t frames = get_arg_val<uint32_t>(5);
    const uint32_t heads = get_arg_val<uint32_t>(6);
    const uint32_t rt_count = get_arg_val<uint32_t>(7);
    const uint32_t row_tile0 = get_arg_val<uint32_t>(8);
    const uint32_t stride_tiles = get_arg_val<uint32_t>(9);

    constexpr bool temporal = get_compile_time_arg_val(0) != 0;
    constexpr bool l2 = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t c_tiles = get_compile_time_arg_val(2);
    constexpr bool rows = get_compile_time_arg_val(3) != 0;
    static_assert(!(rows && temporal), "the rows input has no temporal taps");
    constexpr auto y_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto ones_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer y_buf(cb_y);
    DataflowBuffer w_buf(cb_w);
    DataflowBuffer ones_buf(cb_ones);
    const auto y = TensorAccessor(y_args, y_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto ones = TensorAccessor(ones_args, ones_addr);
    const uint32_t y_bytes = y_buf.get_entry_size();

    if constexpr (temporal) {
        const uint32_t w_bytes = w_buf.get_entry_size();
        w_buf.reserve_back(TAPS * c_tiles);
        for (uint32_t t = 0; t < TAPS * c_tiles; ++t) {
            noc.async_read(w, w_buf, w_bytes, {.page_id = t}, {.offset_bytes = t * w_bytes});
        }
        noc.async_read_barrier();
        w_buf.push_back(TAPS * c_tiles);
    }
    if constexpr (l2) {
        ones_buf.reserve_back(1);
        noc.async_read(ones, ones_buf, ones_buf.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        ones_buf.push_back(1);
    }

    const uint32_t frame_tiles = heads * rt_count * HEAD_TILES;
    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t f = u / (heads * rt_count);
        const uint32_t in_frame = (u % (heads * rt_count)) * HEAD_TILES;
        if constexpr (rows) {
            const uint32_t head = (u / rt_count) % heads;
            const uint32_t row_tile = row_tile0 + f * stride_tiles + u % rt_count;
            y_buf.reserve_back(HEAD_TILES);
            for (uint32_t c = 0; c < HEAD_TILES; ++c) {
                noc.async_read(
                    y,
                    y_buf,
                    y_bytes,
                    {.page_id = row_tile * c_tiles + head * HEAD_TILES + c},
                    {.offset_bytes = c * y_bytes});
            }
            noc.async_read_barrier();
            y_buf.push_back(HEAD_TILES);
            continue;
        }
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            for (uint32_t dt = 0; dt < TAPS; ++dt) {
                const int32_t src = temporal ? static_cast<int32_t>(f + dt) - 2 : static_cast<int32_t>(f);
                if (!temporal && dt > 0) {
                    break;
                }
                if (src < 0 || src >= static_cast<int32_t>(frames)) {
                    continue;
                }
                y_buf.reserve_back(1);
                noc.async_read(
                    y,
                    y_buf,
                    y_bytes,
                    {.page_id = static_cast<uint32_t>(src) * frame_tiles + in_frame + c},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                y_buf.push_back(1);
            }
        }
    }
}
