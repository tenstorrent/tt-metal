// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Sinlge threaded counterparts of existing multithreaded functions
// are differentiated with an extra _ (underscore)
#include "compute_kernel_api/eltwise_binary_single_thread.h"
#include "dataflow_api.h"

namespace NAMESPACE {
void MAIN {
    // Args for computing the results
    // How many blocks of tiles to work on
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);

    // How many tiles per block; needs to be less than
    // circular buffer capacity.
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    // Input and output circular buffer ids.
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Initialize the parts that are common among binary operations
    binary_op_init_common_(cb_in0, cb_in1, cb_out0);

    // Initialize the parts that required specifically for this binary operatoins
    binary_tiles_init_<false, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);

    for (uint32_t block = 0; block < per_core_block_cnt; block++) {
        // Wait for the input circular buffers to be filled with per_core_block_size tiles
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);

        // Perform the elementwise operation on the tiles in the block
        // and store them in the destination register
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            add_tiles_(cb_in0, cb_in1, i, i, i);
        }

        // Wait for enough space to be available in the output circular buffer
        cb_reserve_back_(cb_out0, per_core_block_size);

        // Pack all the output tiles from destination register out to
        // the output circular buffer that resides in L1 memory
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile_(i, cb_out0);
        }

        // Update the write pointer and counts for the output circular buffer.
        cb_push_back_(cb_out0, per_core_block_size);

        // Pop out the used input tiles
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
    }
}
}  // namespace NAMESPACE
