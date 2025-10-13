// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// Writer that uses TensorAccessor to write tiles from c_16 to DRAM
// Compile-time args: TensorAccessorArgs for output tensor (one pack)
// Runtime args:
//   0: out_dram_addr
//   1: num_tiles
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    const uint32_t tile_bytes = get_tile_size(cb_id_out0);

    // Build TensorAccessor from compile-time args
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_id_out0, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(/*tile_idx=*/i, out_acc, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
