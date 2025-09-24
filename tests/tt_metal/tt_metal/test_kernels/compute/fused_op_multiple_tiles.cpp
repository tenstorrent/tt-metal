// if you can't find it: reduce_h4.cpp - DO NOT REMOVE THIS COMMENT

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include "compute_kernel_api/fused_eltwise_binary_reduce.h"
#include "compute_kernel_api/fused_eltwise_binary_reduce_multiple_tiles.h"

#include "/localdev/vbabic/tt-metal/tt_metal/hw/inc/debug/dprint_tensix.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    // Hardware constraint: face_cnt must be <= 8 due to DEST register capacity limit
    // The destination register can hold a maximum of 8 tiles for fused operations
    uint32_t tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input
    constexpr auto cb_inp0 = cb_in0;           // alias for clarity
    constexpr auto cb_inp1 = cb_in1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // output

    cb_wait_front(cb_inp0, tile_cnt);
    cb_wait_front(cb_inp1, tile_cnt);
    cb_reserve_back(cb_out0, 1);

    fused_eltwise_binary_init<ELTWISE_OP_TYPE>(cb_inp0, cb_in1);

    tile_regs_acquire();

    for (uint32_t tile_idx = 0; tile_idx < tile_cnt; ++tile_idx) {
        // the unpacker here calculates the address of the tile in the cb. unpacks the entire tile.
        fused_eltwise_binary_compute<ELTWISE_OP_TYPE>(cb_inp0, cb_inp1, tile_idx, tile_idx, tile_idx);
    }

    // dprint_tensix_dest_reg(0); // - all of dest is filled with expected data

    fused_eltwise_binary_reuse_dest_multiple_tiles(0);
    fused_reduce_populate_ones();

    fused_reduce_init<REDUCE_OP, REDUCE_DIM>();

    int reduce_dst_idx = 0;

    for (uint32_t tile_idx = 0; tile_idx < tile_cnt; ++tile_idx) {
        // For tiles after the first one, we need to reuse the destination as input for that specific tile
        if (tile_idx != 0) {
            fused_eltwise_binary_reuse_dest_multiple_tiles(tile_idx);
        }
        // else{
        //     dprint_tensix_dest_reg(0); // - prints out ones, except for the first row (16 datums)
        // }

        // if(tile_idx == 0){
        //     while(true){} // for debugging srcA - tt-exalens - dr 0,0 srca
        // }

        // Perform the reduce operation on the current tile
        fused_reduce_compute<REDUCE_OP, REDUCE_DIM>(reduce_dst_idx);

        // if(tile_idx == 0){
        //     dprint_tensix_dest_reg(tile_idx);
        // }
    }

    // math hanga nakon compute-a

    // dprint_tensix_dest_reg(0); // - does not hang

    fused_reduce_clear_dvalid_after_for_loop();

    // dprint_tensix_dest_reg(0);    // does not hang

    tile_regs_commit();  // math thread commits the dst register
    tile_regs_wait();    // pack thread waits for the math thread to commit the dst register

    cb_pop_front(cb_inp0, tile_cnt);  // Done with first input
    cb_pop_front(cb_inp1, tile_cnt);  // Done with second input

    // pack tiles
    pack_tile(reduce_dst_idx, cb_out0);

    PACK(for (uint32_t i = 0; i < 32; ++i) { TTI_NOP; });  // stall the packer bcs of the dprint

    // DPRINT_PACK({ DPRINT << "After pack" << ENDL(); });

    DPRINT_PACK({
        DPRINT << "Output tile in cb_out0:" << ENDL();
        for (uint16_t r = 0; r < 32; ++r) {
            DPRINT << (uint)r << " : "
                   << TileSlice(
                          cb_out0,
                          0,
                          SliceRange{
                              .h0 = (uint8_t)r,
                              .h1 = (uint8_t)(r + 1),
                              .hs = (uint8_t)1,
                              .w0 = (uint8_t)0,
                              .w1 = (uint8_t)32,
                              .ws = (uint8_t)1},
                          true,
                          false)
                   << ENDL();
        }
    });

    cb_push_back(cb_out0, 1);

    tile_regs_release();  // pack thread releases the dst register

    fused_reduce_uninit();

    // DPRINT_MATH({ DPRINT << "After reduce_uninit" << ENDL(); }); // hanga nakon N-tog poziva
    // DPRINT_PACK({ DPRINT << "After reduce_uninit" << ENDL(); }); // takodje
    // DPRINT_UNPACK({ DPRINT << "After reduce_uninit" << ENDL(); }); // takodje
}
}  // namespace NAMESPACE
