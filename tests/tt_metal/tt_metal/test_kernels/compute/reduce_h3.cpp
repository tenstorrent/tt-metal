// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/fused_eltwise_binary_reduce.h"
#include "debug/dprint.h"
#include "/localdev/vbabic/tt-metal/tt_metal/hw/inc/debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    // =============================================================================
    // FUSED KERNEL: ELTWISE BINARY + REDUCE OPERATION (Using Fused API)
    // =============================================================================
    // This kernel combines two operations using the fused API:
    // 1. Eltwise binary operation (using fused_eltwise_binary_* functions)
    // 2. Reduce operation (using fused_reduce_* functions)
    // =============================================================================

    // Arguments from eltwise binary kernel
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // Number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);  // Size of each block
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);           // Accumulate to destination flag

    // Arguments from reduce kernel (compile-time constants)
    constexpr uint32_t Ht = get_compile_time_arg_val(0);  // Height in tiles
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // Width in tiles
    constexpr uint32_t NC = get_compile_time_arg_val(2);  // Number of channels

    // =============================================================================
    // CIRCULAR BUFFER (CB) LAYOUT
    // =============================================================================
    // INPUT CBs (for eltwise binary):
    constexpr auto cb_in0 = tt::CBIndex::c_0;  // First input tensor
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // Second input tensor
    constexpr auto cb_inp0 = cb_in0;           // Alias for clarity
    constexpr auto cb_inp1 = cb_in1;           // Alias for clarity

    // REDUCE-SPECIFIC CB:s
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Final output CB

    // =============================================================================
    // MAIN PROCESSING LOOP - PROCESS SINGLE TILES (Using Fused API)
    // =============================================================================
    // Process one tile at a time through the entire pipeline using the fused API

    uint32_t total_tiles = 1;

    for (uint32_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        // =========================================================================
        // STEP 1: FUSED ELTWISE BINARY INITIALIZATION
        // =========================================================================
        fused_eltwise_binary_init<ELTWISE_OP_TYPE>(cb_inp0, cb_inp1);  // potencijalno treba izbaciti PACKER deo?

        // =========================================================================
        // STEP 2: CB SYNC (Wait for input tiles)
        // =========================================================================
        cb_wait_front(cb_inp0, 1);  // Wait for first input tile, ok
        cb_wait_front(cb_inp1, 1);  // Wait for second input tile, ok

        // =========================================================================
        // STEP 3: FUSED ELTWISE BINARY OPERATION
        // =========================================================================
        tile_regs_acquire();

        // *** FUSED ELTWISE OPERATION ***
        fused_eltwise_binary_compute<ELTWISE_OP_TYPE, 0>(
            cb_inp0, cb_inp1, 0, 0);  // template argument for idst, since it has to be 0 for fused op to work

        // dprint_tensix_dest_reg(0); // prints the correct results!

        // *** FUSED DESTINATION REUSE ***
        // This prepares the destination registers for the reduce operation
        fused_eltwise_binary_reuse_dest();

        // debug prints:
        // DPRINT_MATH({ DPRINT << "debug print math, works after reuse_dest" << ENDL(); }); // -will be printed out
        // DPRINT_PACK({ DPRINT << "debug print pack, works after reuse_dest" << ENDL(); }); // -will be printed out
        // DPRINT_UNPACK({ DPRINT << "debug print unpack, works after reuse_dest" << ENDL(); }); // -will be printed out

        // DON'T release tile_regs here - we need them for the reduce operation!

        // =========================================================================
        // STEP 4: CB SYNC (Clean up eltwise inputs, make intermediate available)
        // =========================================================================
        cb_pop_front(cb_inp0, 1);  // Done with first input
        cb_pop_front(cb_inp1, 1);  // Done with second input

        // =========================================================================
        // STEP 5: FUSED REDUCE INITIALIZATION
        // =========================================================================
        fused_reduce_init<REDUCE_OP, REDUCE_DIM>();
        // DPRINT_MATH({ DPRINT << "After reduce_init" << ENDL(); }); // -will be printed out
        // DPRINT_PACK({ DPRINT << "After reduce_init" << ENDL(); }); // this too
        // DPRINT_UNPACK({ DPRINT << "After reduce_init" << ENDL(); }); // this too

        // =========================================================================
        // STEP 6: FUSED REDUCE OPERATION
        // =========================================================================
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;

        // NOTE: We already have destination registers acquired from the eltwise operation
        // The fused_eltwise_binary_reuse_dest() has prepared the data for reduction

        // dprint_tensix_dest_reg(0); // prints out ones, just as expected!!

        // *** FUSED REDUCE OPERATION ***
        fused_reduce_compute<REDUCE_OP, REDUCE_DIM>(reduce_dst_idx);

        // DPRINT_MATH({ DPRINT << "After reduce_compute" << ENDL(); }); // is being printed out
        // DPRINT_PACK({ DPRINT << "After reduce_compute" << ENDL(); }); // is being printed out
        //  DPRINT_UNPACK({ DPRINT << "After reduce_compute" << ENDL(); }); // is being printed out

        // since math is doing something here, let's see if we have correct results in dest:
        // dprint_tensix_dest_reg(0); // prints out the correct results since math does not hang anymore

        // while(true){} // loop if you want to check the srcA - tt-exalens - dr 0,0 srca

        tile_regs_commit();  // MATH((llk_math_dest_section_done<DST_ACCUM_MODE>())); - releases lock on DST register

        // DPRINT_MATH({ DPRINT << "After reduce_compute" << ENDL(); });
        // DPRINT_PACK({ DPRINT << "After reduce_compute" << ENDL(); });
        // DPRINT_UNPACK({ DPRINT << "After reduce_compute" << ENDL(); });

        // tile_regs_wait();
        acquire_dst();

        // Pack and output the reduced result
        cb_reserve_back(cb_out0, onetile);

        DPRINT_MATH({ DPRINT << "After tile wait" << ENDL(); });    // does not hang
        DPRINT_PACK({ DPRINT << "After tile wait" << ENDL(); });    // does not hang
        DPRINT_UNPACK({ DPRINT << "After tile wait" << ENDL(); });  // does not hang

        pack_tile(reduce_dst_idx, cb_out0);

        DPRINT_PACK({  // - does not hang, prints out garbage data
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

        cb_push_back(cb_out0, onetile);

        // tile_regs_release();  // Finally release the tile registers
        release_dst();
    }

    // =============================================================================
    // CLEANUP PHASE (Using Fused API)
    // =============================================================================
    fused_reduce_uninit();  // Clean up reduce operation resources
}
}  // namespace NAMESPACE
