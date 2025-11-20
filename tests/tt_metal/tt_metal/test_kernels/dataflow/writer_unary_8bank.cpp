// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr{get_arg_val<uint32_t>(0U)};
    const uint32_t num_tiles{get_arg_val<uint32_t>(2U)}; // Index 2 to match with regular writer_unary

    constexpr uint32_t cb_id_out0{16U};
    constexpr uint32_t onetile{1U};
    const uint32_t tile_bytes{get_tile_size(cb_id_out0)};

    constexpr auto dst_args{TensorAccessorArgs<0U>()};
    const auto s{TensorAccessor(dst_args, dst_addr, tile_bytes)};

    for (uint32_t i{0U}; i < num_tiles; ++i) {
        const uint64_t dst_noc_addr{get_noc_addr(i, s)};

        cb_wait_front(cb_id_out0, onetile);
        const uint32_t l1_read_addr{get_read_ptr(cb_id_out0)};

        noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, onetile);
    }
}
