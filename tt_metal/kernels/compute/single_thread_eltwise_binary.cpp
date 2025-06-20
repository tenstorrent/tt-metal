// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary_st.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint.h"

#if 0
#define DUNPACK(x) UNPACK(x)
#else
#define DUNPACK(x)
#endif

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
        UNPACK(( DPRINT << "======" << ENDL() ));
        for (uint8_t r = 0; r < 32; ++ r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            UNPACK(( DPRINT << (uint)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
        }
        UNPACK(( DPRINT << "++++++" << ENDL() ));
}

inline void print_cb_details(uint32_t cb_id) {
    UNPACK(DPRINT<< "cb_id " << cb_id << ": { "
                 << "size: " << get_local_cb_interface(cb_id).fifo_size << ", "
                 << "limit: " << get_local_cb_interface(cb_id).fifo_limit << ", "
                 << "page_size: " << get_local_cb_interface(cb_id).fifo_page_size << ", "
                 << "num_pages: " << get_local_cb_interface(cb_id).fifo_num_pages << ", "
                 << "rd_ptr: " << get_local_cb_interface(cb_id).fifo_rd_ptr << ", "
                 << "wr_ptr: " << get_local_cb_interface(cb_id).fifo_wr_ptr << ", "
                 << "wr_tile_ptr: " << get_local_cb_interface(cb_id).fifo_wr_tile_ptr << " }");
}

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

        DUNPACK(DPRINT << "WFD" << ENDL());

        // Wait for enough space to be available in the output circular buffer
        cb_reserve_back_st(cb_out0, per_core_block_size);

        DUNPACK(DPRINT << "RBD" << ENDL());

        // tile_regs_acquire_st();

        DUNPACK(DPRINT << "RACQD" << ENDL());

        // Perform the elementwise operation on the tiles in the block
        // and store them in the destination register
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            add_tiles_st(cb_in0, cb_in1, i, i, i);
        }

        DUNPACK(DPRINT << "ADDD" << ENDL());

        // tile_regs_commit_st();

        DUNPACK(DPRINT << "RCOMMD" << ENDL());

        // tile_regs_wait_st();

        DUNPACK(DPRINT << "RWD" << ENDL());

	print_cb_details(cb_out0);
        //	print_full_tile(cb_out0);
        // Pack all the output tiles from destination register out to
        // the output circular buffer that resides in L1 memory
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile_st(i, cb_out0);
        }

        DUNPACK(DPRINT << "PACKD" << ENDL());

        // tile_regs_release_st();

        DUNPACK(DPRINT << "RRELD" << ENDL());

        // Update the write pointer and counts for the output circular buffer.
        cb_push_back_st(cb_out0, per_core_block_size);

	print_cb_details(cb_out0);
//	print_full_tile(cb_out0);

        DUNPACK(DPRINT << "PBD" << ENDL());

        // Pop out the used input tiles
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);

        DUNPACK(DPRINT << "POPD" << ENDL());
    }

    UNPACK(DPRINT << "UE" << ENDL());
    PACK(DPRINT << "PE" << ENDL());
    MATH(DPRINT << "ME" << ENDL());
}
}  // namespace NAMESPACE
