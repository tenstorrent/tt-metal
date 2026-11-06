// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

constexpr bool stream_input = get_compile_time_arg_val(0);

void kernel_main() {
    DataflowBuffer input(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    if constexpr (!stream_input) {
        input.reserve_back(num_tiles);
        input.push_back(num_tiles);
        return;
    }

    // Feed the resident input tensor through the smaller compute input CB one tile at a time, so the
    // compute read pointer wraps inside a single reduction.
    DataflowBuffer source(3);
    const uint32_t tile_bytes = get_tile_size(0);
    const uint32_t source_base = source.get_read_ptr();
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        input.reserve_back(1);
        noc_async_read(get_noc_addr(source_base + tile * tile_bytes), input.get_write_ptr(), tile_bytes);
        noc_async_read_barrier();
        input.push_back(1);
    }
}
