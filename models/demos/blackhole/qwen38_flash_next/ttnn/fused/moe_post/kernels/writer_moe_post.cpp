// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE post program, writer: the local sum's column tile into the reduce-scatter's input.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_col = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto out_args = TensorAccessorArgs<0, 0>();
    const auto out = TensorAccessor(out_args, out_addr);

    const Noc noc;
    DataflowBuffer out_tile(cb_out);
    {
        out_tile.wait_front(1);
        noc.async_write(out_tile, out, tile_bytes, {}, {.page_id = tile_col});
        noc.async_write_barrier();
    }
    out_tile.pop_front(1);
}
