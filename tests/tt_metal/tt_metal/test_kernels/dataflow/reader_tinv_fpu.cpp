// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

// Reads the identity tile (c_1), the three quadrant masks (c_2) and num_tiles negN tiles (c_0), one DRAM page each.
// Runtime args: 0 negN addr, 1 negN bank, 2 eye addr, 3 eye bank, 4 masks addr, 5 masks bank, 6 num_tiles.
void kernel_main() {
    const uint32_t n_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_bank = get_arg_val<uint32_t>(1);
    const uint32_t eye_addr = get_arg_val<uint32_t>(2);
    const uint32_t eye_bank = get_arg_val<uint32_t>(3);
    const uint32_t mask_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_bank = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles = get_arg_val<uint32_t>(6);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    CircularBuffer cb_n(0);
    CircularBuffer cb_eye(1);
    CircularBuffer cb_mask(2);
    const uint32_t tile_bytes = cb_n.get_tile_size();

    cb_eye.reserve_back(1);
    noc.async_read(dram, cb_eye, tile_bytes, {.bank_id = eye_bank, .addr = eye_addr}, {});
    noc.async_read_barrier();
    cb_eye.push_back(1);

    cb_mask.reserve_back(3);
    noc.async_read(dram, cb_mask, 3 * tile_bytes, {.bank_id = mask_bank, .addr = mask_addr}, {});
    noc.async_read_barrier();
    cb_mask.push_back(3);

    uint32_t addr = n_addr;
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_n.reserve_back(1);
        noc.async_read(dram, cb_n, tile_bytes, {.bank_id = n_bank, .addr = addr}, {});
        noc.async_read_barrier();
        cb_n.push_back(1);
        addr += tile_bytes;
    }
}
