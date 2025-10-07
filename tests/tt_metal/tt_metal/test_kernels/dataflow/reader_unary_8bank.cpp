// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"

void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = 2;
    uint32_t scaler = get_arg_val<uint32_t>(8);
    union {
        float f;
        uint32_t u;
    } u;
    u.u = scaler;
    // DPRINT << "basic Scaler = " << F32(u.f) << ENDL();
    cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(0);
    }

    for (int k = 0; k < 4; k++) {
        for (int j = 0; j < 16; j++) {
            ptr[k * 256 + j] = uint16_t(u.u >> 16);
        }
    }
    cb_push_back(cb_in_2, 1);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);  // Number of rows (height in tiles)
    uint32_t Wt = get_arg_val<uint32_t>(2);  // Number of cols (width in tiles)
    uint32_t NC = get_arg_val<uint32_t>(3);  // Number of channels

    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t log_2_tile_bytes = 11;  // log2(2048) = 11

    // Use bank_id 0 as default for interleaved buffer
    constexpr uint32_t bank_id = 0;

#if GENERATE_BCAST_SCALER
    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    constexpr uint32_t blk = BLOCK_SIZE;
#else
    constexpr uint32_t blk = 1;  // 1 for correctness for unfused kernels
#endif

#ifdef TILE_OFFSET
    uint32_t tile_offset = TILE_OFFSET;
#else
    constexpr uint32_t tile_offset = 0;
#endif
    // DPRINT << "Reader Tile offset=" << tile_offset << ENDL();

    // Read all Ht*Wt tiles for all NC channels at once (like reduce_c pattern)
    uint32_t total_tiles = NC * Ht * Wt;
    cb_reserve_back(cb_id_in0, total_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                uint32_t tile_idx = nc * Ht * Wt + ht * Wt + wt;
                uint64_t src_noc_addr =
                    get_noc_addr_from_bank_id<true>(bank_id, src_addr + (tile_idx + tile_offset) * tile_bytes);
                auto addr = l1_write_addr + (tile_idx << log_2_tile_bytes);
                noc_async_read(src_noc_addr, addr, tile_bytes);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, total_tiles);
}
