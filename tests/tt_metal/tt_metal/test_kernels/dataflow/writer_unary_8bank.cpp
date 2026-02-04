// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(2); // Index 2 to match with regular writer_unary

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out0);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    experimental::CircularBuffer cb(cb_id_out0);
    experimental::Noc noc;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb.wait_front(onetile);
        noc.async_write(cb, s, tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(onetile);
    }
}
