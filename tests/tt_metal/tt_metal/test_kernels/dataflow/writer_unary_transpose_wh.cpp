// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    //uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t N = get_arg_val<uint32_t>(4);
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t HtWtTileBytes = HtWt*2048; // TODO(AP): assumed 16-bits
    uint32_t WtTileBytes = Wt*2048; // TODO(AP): assumed 16-bits

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    uint32_t dst_addrN = dst_addr;
    // this writer will write a NWH tensor in NHW order
    for (uint32_t n = 0; n<N; n++) {
        dst_addr = dst_addrN;
        for (uint32_t w = 0; w<Wt; w++) {
            for (uint32_t h = 0; h<Ht; h++) {
                uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);
                cb_wait_front(cb_id_out0, ublock_size_tiles);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
                noc_async_write_barrier();

                cb_pop_front(cb_id_out0, ublock_size_tiles);
                dst_addr += WtTileBytes; // stride in H
            } // Ht
            dst_addr -= HtWtTileBytes; // go back to H=0
            dst_addr += ublock_size_bytes; // increment Wt
        } // Wt
        dst_addrN += HtWtTileBytes;
    } // N
}
