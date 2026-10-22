// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    CircularBuffer cb_out(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr);
    constexpr uint32_t out_tile_size = get_tile_size(cb_id_out);

    uint32_t tile_id = start_id;
    for (uint32_t done = 0; done < num_tiles; done += block_tiles) {
        const uint32_t n = (num_tiles - done) < block_tiles ? (num_tiles - done) : block_tiles;
        cb_out.wait_front(n);
        uint32_t l1_read_addr = cb_out.get_read_ptr();
        for (uint32_t t = 0; t < n; ++t) {
            noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, out_tile_size, {}, {.page_id = tile_id});
            l1_read_addr += out_tile_size;
            ++tile_id;
        }
        noc.async_writes_flushed();
        cb_out.pop_front(n);
    }
    noc.async_write_barrier();
}
