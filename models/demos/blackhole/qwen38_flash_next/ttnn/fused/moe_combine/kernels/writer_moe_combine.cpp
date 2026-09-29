// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE combine program, writer: the unit's `cols` output tiles (row tile r, column tiles g * cols ..) into the tiled
// [1, 1, rows, 2560] output, page id = r * hidden_tiles + column tile; the writes of a unit are issued as the compute
// packs them and barriered once per unit.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    if (unit_count == 0) {
        return;
    }

    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cols = get_named_compile_time_arg_val("cols");
    constexpr uint32_t groups = get_named_compile_time_arg_val("groups");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t tile_bytes = 2048;

    constexpr auto out_args = TensorAccessorArgs<0, 0>();
    const auto out = TensorAccessor(out_args, out_addr);

    DataflowBuffer out_tiles(cb_out);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        FUSED_ZONE("fz_mc_w_unit");
        const uint32_t r = u / groups;
        const uint32_t g = u % groups;
        const uint32_t first = r * hidden_tiles + g * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            out_tiles.wait_front(c + 1);  // the tiles of a unit never straddle the ring's end (2 x cols pages)
            noc_async_write(out_tiles.get_read_ptr() + c * tile_bytes, out.get_noc_addr(first + c), tile_bytes);
        }
        noc_async_write_barrier();
        out_tiles.pop_front(cols);
    }
}
