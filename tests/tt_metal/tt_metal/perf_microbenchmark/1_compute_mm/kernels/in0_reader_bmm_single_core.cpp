// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
// #include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t in0_block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_in0 = 0;

    //    DPRINT  << TSLICE(cb_id_in0, 0, SliceRange::h0_w0_32()) << ENDL() ;

    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb_id_in0, in0_block_tiles);
        cb_push_back(cb_id_in0, in0_block_tiles);
    }
}
