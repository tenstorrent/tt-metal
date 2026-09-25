// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One tile of the tile clock network, on whichever RISC the tile has. The tile announces itself with the host's
// nonce, waits for its turn, reads each partner tile's wall clock over the NoC the host named (`reps` brackets
// each) and writes per partner the median and the quartile spread of 2 * (partner wall - bracket midpoint) in the
// clocks' low words, the median round trip, and one coarse whole-clock difference that says which 2^32-tick turn
// the median sits in, then announces done with the nonce inverted, waits to be released and zeroes its scratch.
// Partners share this tile's row or column, NoC 0 towards higher coordinates and NoC 1 towards lower, so a pair's
// two readings cross the same links in opposite directions.
//
// Arguments: scratch address, reps, nonce, partner count, then a partner per argument as noc << 31 | y << 16 | x.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "tools/profiler/sync/tile_read.hpp"

namespace kp = kernel_profiler;

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t reps = get_arg_val<uint32_t>(1);
    const uint32_t nonce = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr kp::TileNetScratch* s = reinterpret_cast<volatile tt_l1_ptr kp::TileNetScratch*>(scratch);
    volatile tt_l1_ptr kp::TileNetTable& tab = s->table;
    tab.go = 0;
    tab.ready = nonce;
    uint32_t go;
    do {
        invalidate_l1_cache();
        go = tab.go;
    } while (go == 0);
    if (go == kp::kTileNetGoMeasure) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t w = get_arg_val<uint32_t>(4 + k);
            tile_read::measure(w >> 31, tile_read::coord(w & 0x7FFFFFFFu), scratch, reps, s->hist, tab.partner[k]);
        }
        tab.ready = ~nonce;
        do {
            invalidate_l1_cache();
        } while (tab.go != kp::kTileNetGoExit);
    }
    // On a dispatch core the scratch is the profiler's ring space, which its first frame must find zero.
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < sizeof(kp::TileNetScratch) / 4; i++) {
        p[i] = 0;
    }
}
