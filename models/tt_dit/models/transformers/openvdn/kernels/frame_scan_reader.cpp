// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for frame_scan_compute.cpp. T and J are `[F, H, 128, 128]` fp32 tiled: matrix (f, head) is pages
// 16 (f H + head) .. + 15, row block r its pages 16 (f H + head) + 4 r .. + 3. The start state is `[1, H, 128, 128]`.
// Per step (frame f in scan order): T[f]'s 16 tiles (not for the first frame without a start state), then this block's
// 4 tiles of J[f].
//
// Compile-time args: has_start, then the tensor accessor args of T, J, start.
// Runtime args: T, J, start addresses, first unit, number of units, frames, heads. Unit u is head u / 8, reverse
// (u / 4) % 2, row block u % 4.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_t = 0;
constexpr uint32_t cb_j = 1;
constexpr uint32_t cb_start = 2;
constexpr uint32_t TILES = 16;
constexpr uint32_t NT = 4;
}  // namespace

void kernel_main() {
    const uint32_t t_addr = get_arg_val<uint32_t>(0);
    const uint32_t j_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t frames = get_arg_val<uint32_t>(5);
    const uint32_t heads = get_arg_val<uint32_t>(6);

    constexpr bool has_start = get_compile_time_arg_val(0) != 0;
    constexpr auto t_args = TensorAccessorArgs<1>();
    constexpr auto j_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto start_args = TensorAccessorArgs<j_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer t_buf(cb_t);
    DataflowBuffer j_buf(cb_j);
    DataflowBuffer start_buf(cb_start);
    const uint32_t tile_bytes = t_buf.get_entry_size();
    const auto t_src = TensorAccessor(t_args, t_addr);
    const auto j_src = TensorAccessor(j_args, j_addr);
    const auto start_src = TensorAccessor(start_args, start_addr);

    const auto read_tile = [&](DataflowBuffer& buf, const auto& src, uint32_t page) {
        buf.reserve_back(1);
        noc.async_read(src, buf, tile_bytes, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        buf.push_back(1);
    };

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t head = u / (2 * NT), reverse = (u / NT) % 2, block = u % NT;
        if constexpr (has_start) {
            for (uint32_t j = 0; j < NT; ++j) {
                read_tile(start_buf, start_src, head * TILES + block * NT + j);
            }
        }
        for (uint32_t step = 0; step < frames; ++step) {
            const uint32_t f = reverse ? frames - 1 - step : step;
            const uint32_t matrix = (f * heads + head) * TILES;
            if (has_start || step > 0) {
                for (uint32_t t = 0; t < TILES; ++t) {
                    read_tile(t_buf, t_src, matrix + t);
                }
            }
            for (uint32_t j = 0; j < NT; ++j) {
                read_tile(j_buf, j_src, matrix + block * NT + j);
            }
        }
    }
}
