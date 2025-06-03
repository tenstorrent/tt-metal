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
    // Args for reading data from DRAM
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);

    // Args for computing the results
    // How many blocks of tiles to work on
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(4);

    // How many tiles per block; needs to be less than
    // circular buffer capacity.
    uint32_t per_core_block_size = get_arg_val<uint32_t>(5);

    // For writing out the results
    uint32_t dst_addr = get_arg_val<uint32_t>(6);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(7);

    // Input and output circular buffer ids.
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_in1);
    uint32_t ublock_size_bytes_dst = get_tile_size(cb_out0);

    // Initialize the parts that are common among binary operations
    binary_op_init_common_(cb_in0, cb_in1, cb_out0);

    // Initialize the parts that required specifically for this binary operatoins
    binary_tiles_init_<false, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);

    for (uint32_t block = 0; block < per_core_block_cnt; block++) {
        cb_push_back_from_dram(src0_bank_id, src0_addr, cb_in0, per_core_block_size);
        src0_addr += ublock_size_bytes_0 * per_core_block_size;

        cb_push_back_from_dram(src1_bank_id, src1_addr, cb_in1, per_core_block_size);
        src1_addr += ublock_size_bytes_1 * per_core_block_size;

        // Perform the elementwise operation on the tiles in the block
        // and store them in the destination register
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            add_tiles_(cb_in0, cb_in1, i, i, i);
        }

        // Pack all the output tiles from destination register out to
        // the output circular buffer that resides in L1 memory
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile_(i, cb_out0);
        }

        // Update the write pointer and counts for the output circular buffer.
        cb_push_back_(cb_out0, per_core_block_size);
        cb_pop_front_to_dram(dst_bank_id, dst_addr, cb_out0, per_core_block_size);
        dst_addr += ublock_size_bytes_dst * per_core_block_size;

        // Pop out the used input tiles
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
    }
}
}  // namespace NAMESPACE
