// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(4);
    uint32_t src1_noc_y = get_arg_val<uint32_t>(5);
    // skip arg 7 for compat with reader_diff_lengths
    uint32_t NCHtWt = get_arg_val<uint32_t>(6);
    uint32_t NC = get_arg_val<uint32_t>(7);
    uint32_t Ht = get_arg_val<uint32_t>(8);
    uint32_t Wt = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);
    experimental::Noc noc;
    uint32_t tile_bytes = cb_in0.get_tile_size();

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i1 = 0;
    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            {
                // only read one tile in H per W-line of tiles
                // So we push a total of NC*H tiles from src1
                cb_in1.reserve_back(onetile);
                noc.async_read(
                    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
                    cb_in1,
                    tile_bytes,
                    {.bank_id = src1_bank_id, .addr = src1_addr},
                    {});
                noc.async_read_barrier();
                cb_in1.push_back(onetile);
                src1_addr += tile_bytes;
            }

            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_in0.reserve_back(onetile);
                noc.async_read(
                    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
                    cb_in0,
                    tile_bytes,
                    {.bank_id = src0_bank_id, .addr = src0_addr},
                    {});
                noc.async_read_barrier();
                cb_in0.push_back(onetile);
                src0_addr += tile_bytes;
            }  // Wt loop
        }  // Ht loop
        src1_addr = get_arg_val<uint32_t>(4);  // reset the H-tile ptr
    }  // NC loop
}
