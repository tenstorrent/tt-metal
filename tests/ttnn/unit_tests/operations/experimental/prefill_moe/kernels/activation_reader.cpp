// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute V0: Activation reader kernel (RISCV_1 / Reader)
// Reads activation tiles [1, K_tiles] from interleaved DRAM → CB_ACT
// One K-tile at a time, synchronized with compute via CB flow control.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_k_tiles = get_arg_val<uint32_t>(1);
    const uint32_t act_start_tile = get_arg_val<uint32_t>(2);

    // Compile-time args: TensorAccessorArgs for activation tensor
    constexpr auto act_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_act = 0;

    const uint32_t page_bytes = get_local_cb_interface(cb_act).fifo_page_size;
    const auto act_accessor = TensorAccessor(act_args, act_addr, page_bytes);

    uint32_t tile_id = act_start_tile;

    for (uint32_t k = 0; k < num_k_tiles; ++k) {
        cb_reserve_back(cb_act, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_act);
        noc_async_read_page(tile_id, act_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_act, 1);
        tile_id++;
    }
}
