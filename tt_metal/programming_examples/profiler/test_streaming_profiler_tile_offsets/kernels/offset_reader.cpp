// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Resident on one idle eth core per chip. At each checkpoint the host asks for, it reads every target tile's wall
// clock over NoC 0 with the tile clock network's estimator (tools/profiler/sync/tile_read.hpp) into
// offset_reader::Scratch. Between checkpoints it reads nothing off its own tile.
//
// Arguments: scratch address, target count, reps.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/sync/tile_read.hpp"
#include "offset_reader.hpp"

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t reps = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr offset_reader::Scratch* s =
        reinterpret_cast<volatile tt_l1_ptr offset_reader::Scratch*>(scratch);
    for (;;) {
        uint32_t go;
        do {
            invalidate_l1_cache();
            go = s->go;
        } while (go == s->done);
        if (go == offset_reader::kExit) {
            break;
        }
        for (uint32_t k = 0; k < n; k++) {
            tile_read::measure(0, tile_read::coord(s->targets[k]), scratch, reps, s->hist, s->out[k]);
        }
        s->done = go;
    }
}
