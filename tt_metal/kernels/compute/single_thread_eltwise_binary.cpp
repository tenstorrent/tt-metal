// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary_st.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

    // How many blocks of tiles to work on
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    
    // How many tiles per block
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    // Input and output circular buffer ids.
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Initialize the parts that are common among binary operations
    binary_op_init_common_st(cb_in0, cb_in1, cb_out0);

    // Initialize the parts that required specifically for this binary operatoins
    binary_tiles_init_st<false, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
	// Wait for the input circular buffers to be filled with per_core_block_size tiles
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);

        UNPACK(DPRINT<< "WFD" << ENDL());	

	// Wait for enough space to be available in the output circular buffer
        cb_reserve_back_st(cb_out0, per_core_block_size);

        UNPACK(DPRINT <<"RBD" <<ENDL());	

        tile_regs_acquire_st();
        
        UNPACK(DPRINT <<"RACQD" <<ENDL());	

	// Perform the elementwise operation on the tiles in the block 
	// and store them in the destination register
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            add_tiles_st(cb_in0, cb_in1, i, i, i);
        }

        UNPACK(DPRINT <<"ADDD" <<ENDL());	

        tile_regs_commit_st();

        UNPACK(DPRINT <<"RCOMMD" <<ENDL());	

        tile_regs_wait_st();

        UNPACK(DPRINT <<"RWD" <<ENDL());	

        // Pack all the output tiles from destination register out to 
	// the output circular buffer that resides in L1 memory	
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile_st(i, cb_out0);
        }

        UNPACK(DPRINT <<"PACKD" <<ENDL());	

        tile_regs_release_st();

        UNPACK(DPRINT <<"RRELD" <<ENDL());	

	// Update the write pointer and counts for the output circular buffer. 
        cb_push_back_st(cb_out0, per_core_block_size);

        UNPACK(DPRINT <<"PBD" <<ENDL());	

	// Pop out the used input tiles
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);

        UNPACK(DPRINT <<"POPD" <<ENDL());	
    }

    UNPACK(DPRINT << "UE" << ENDL());
    //PACK(DPRINT << "PE" << ENDL());
    //MATH(DPRINT << "ME" << ENDL());
}
}  // namespace NAMESPACE
