// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
// #include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t in0_block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_in0 = 0;

    // DPRINT("{}\n", TSLICE(cb_id_in0, 0, SliceRange::h0_w0_32()));

#ifdef FUSE_BIAS
    // Bias equivalence test: the bias CB (c_3) is a globally-allocated CB pre-populated
    // by the host, so this reader only has to advance its producer pointer once so the
    // compute kernel's cb_wait_front(bias_cb_id, in1_per_core_w) sees the tiles. Nothing
    // pops it (matches the copy kernel, which holds bias resident for the whole run).
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_bias = 3;
    cb_reserve_back(cb_id_bias, bias_ntiles);
    cb_push_back(cb_id_bias, bias_ntiles);
#endif

    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb_id_in0, in0_block_tiles);
        cb_push_back(cb_id_in0, in0_block_tiles);
    }
}
