// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t src_dram_bank_id = get_arg_val<uint32_t>(1);
    // uint32_t unused  = get_arg_val<uint32_t>(2);
    // uint32_t unused  = get_arg_val<uint32_t>(3);
    // skip 3 for compat with reader_unary_8bank, reader_unary
    uint32_t N = get_arg_val<uint32_t>(4);
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t HtWtTileBytes = HtWt * 2048;  // TODO(AP): assumed 16-bits
    uint32_t WtTileBytes = Wt * 2048;      // TODO(AP): assumed 16-bits

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t src_addrN = src_addr;
    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        src_addr = src_addrN;
        for (uint32_t w = 0; w<Wt; w++) {
            for (uint32_t h = 0; h<Ht; h++) {
                uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_dram_bank_id, src_addr);
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
                noc_async_read_barrier();

                cb_push_back(cb_id_in0, onetile);
                src_addr += WtTileBytes;  // stride in H
            }  // Ht
            src_addr -= HtWtTileBytes;  // go back to H=0
            src_addr += tile_bytes;     // increment Wt
        }  // Wt
        src_addrN += HtWtTileBytes;
    }  // N
}
