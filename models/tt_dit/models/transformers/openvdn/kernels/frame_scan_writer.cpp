// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for frame_scan_compute.cpp: per unit and step (frame f in scan order) the unit's 4 state tiles, to pages
// 16 (f H + head) + 4 r .. + 3 of the prefix (forward) or suffix (reverse) bank, `[F, H, 128, 128]` fp32 tiled.
//
// Compile-time args: the tensor accessor args of the banks (both have the same spec).
// Runtime args: prefix, suffix addresses, first unit, number of units, frames, heads. Unit u is head u / 8, reverse
// (u / 4) % 2, row block u % 4.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t TILES = 16;
constexpr uint32_t NT = 4;
}  // namespace

void kernel_main() {
    const uint32_t prefix_addr = get_arg_val<uint32_t>(0);
    const uint32_t suffix_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t frames = get_arg_val<uint32_t>(4);
    const uint32_t heads = get_arg_val<uint32_t>(5);

    constexpr auto out_args = TensorAccessorArgs<0>();

    Noc noc;
    DataflowBuffer out_buf(cb_out);
    const uint32_t tile_bytes = out_buf.get_entry_size();
    const auto prefix = TensorAccessor(out_args, prefix_addr);
    const auto suffix = TensorAccessor(out_args, suffix_addr);

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t head = u / (2 * NT), reverse = (u / NT) % 2, block = u % NT;
        for (uint32_t step = 0; step < frames; ++step) {
            const uint32_t f = reverse ? frames - 1 - step : step;
            const uint32_t first_page = (f * heads + head) * TILES + block * NT;
            for (uint32_t j = 0; j < NT; ++j) {
                out_buf.wait_front(1);
                if (reverse) {
                    noc.async_write(out_buf, suffix, tile_bytes, {}, {.page_id = first_page + j});
                } else {
                    noc.async_write(out_buf, prefix, tile_bytes, {}, {.page_id = first_page + j});
                }
                noc.async_writes_flushed();
                out_buf.pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
}
