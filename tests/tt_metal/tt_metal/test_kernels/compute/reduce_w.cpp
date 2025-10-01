// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint_tensix.h"
#include "ckernel.h"
#include "hw/inc/wormhole/dev_mem_map.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

    // Like reduce_c: wait on all tiles at once, then process in double loop
    constexpr uint32_t total_tiles = NC * Ht * Wt;
    constexpr uint32_t output_tiles = NC * Ht;  // One output tile per row per channel

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
    reduce_max_row_init();
    // reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);

    // Wait on scaler and all input tiles at once (like reduce_c pattern)
    cb_wait_front(tt::CBIndex::c_2, 1);  // scaler tile from the reader
    cb_wait_front(tt::CBIndex::c_0, total_tiles);      // all input tiles
    cb_reserve_back(tt::CBIndex::c_16, output_tiles);  // reserve space for all outputs

    constexpr uint32_t reduce_dst_idx = 0;

    // Synchronize UNPACK, MATH and PACK threads before timing
    volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;

    UNPACK((base_address[1] = 1));
    MATH((base_address[2] = 2));
    PACK((base_address[3] = 3));
    while (base_address[1] != 1) {
        asm("nop");
    }
    while (base_address[2] != 2) {
        asm("nop");
    }
    while (base_address[3] != 3) {
        asm("nop");
    }
    UNPACK((base_address[5] = 5));
    MATH((base_address[6] = 6));
    PACK((base_address[7] = 7));
    while (base_address[5] != 5) {
        asm("nop");
    }
    while (base_address[6] != 6) {
        asm("nop");
    }
    while (base_address[7] != 7) {
        asm("nop");
    }
    UNPACK((base_address[1] = 0));
    MATH((base_address[2] = 0));
    PACK((base_address[3] = 0));
    while (base_address[1] != 0) {
        asm("nop");
    }
    while (base_address[2] != 0) {
        asm("nop");
    }
    while (base_address[3] != 0) {
        asm("nop");
    }
    UNPACK((base_address[5] = 0));
    MATH((base_address[6] = 0));
    PACK((base_address[7] = 0));

    // Only PACK thread measures timing to avoid interference
    uint64_t start_time = 0;
    UNPACK((start_time = ckernel::read_wall_clock()));

    // Double loop pattern like reduce_c: outer loop over channels and rows, inner loop over cols
    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            acquire_dst();
            // Reduce across W dimension (cols) for this row
            for (uint32_t wt = 0; wt < Wt; wt++) {
                uint32_t tile_idx = nc * Ht * Wt + ht * Wt + wt;
                reduce_tile_max_row(tt::CBIndex::c_0, tt::CBIndex::c_2, tile_idx, reduce_dst_idx);
                // reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
            }
            // Pack result to output CB (implicit pack_tile behavior like reduce_c)
            uint32_t output_idx = nc * Ht + ht;
            pack_tile(reduce_dst_idx, tt::CBIndex::c_16);
            release_dst();
        }
    }

    // Only UNPACK thread measures end timing and prints results
    uint64_t end_time = 0;
    uint64_t elapsed_time = 0;
    UNPACK((end_time = ckernel::read_wall_clock()));
    UNPACK((elapsed_time = end_time - start_time));

    // Print timing information (only on UNPACK thread)
    UNPACK((DPRINT << "Reduce loop timing: " << elapsed_time << " cycles" << ENDL()));
    UNPACK((DPRINT << "Total tiles processed: " << (NC * Ht * Wt) << ENDL()));
    UNPACK((DPRINT << "Cycles per tile: " << (elapsed_time / (NC * Ht * Wt)) << ENDL()));

    // Push all outputs at once
    cb_push_back(tt::CBIndex::c_16, output_tiles);
    reduce_uninit();
}
}  // namespace NAMESPACE
