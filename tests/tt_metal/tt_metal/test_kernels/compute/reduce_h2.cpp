// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce.h"
#include "/localdev/vbabic/tt-metal/tt_metal/hw/inc/debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    // =============================================================================
    // FUSED KERNEL: ELTWISE BINARY + REDUCE OPERATION
    // =============================================================================
    // This kernel combines two operations:
    // 1. Eltwise binary operation (from eltwise_binary.cpp)
    // 2. Reduce operation (from reduce_h.cpp)
    //
    // Data flow: cb_in0, cb_in1 -> ELTWISE_OP -> cb_intermediate -> REDUCE_OP -> cb_out0
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

    // INTERMEDIATE CB (connects eltwise -> reduce):
    constexpr auto cb_intermediate = tt::CBIndex::c_24;  // Output of eltwise, input to reduce

    // REDUCE-SPECIFIC CBs:
    constexpr auto cb_in2 = tt::CBIndex::c_2;    // Scaler tile for reduce operation
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Final output CB

    // =============================================================================
    // INITIALIZATION PHASE
    // =============================================================================

    // Initialize eltwise binary operation (sets up ALU for binary ops)
    binary_op_init_common(cb_inp0, cb_inp1, cb_intermediate);

    // Initialize binary tiles (conditional compilation from original eltwise_binary.cpp)
    binary_tiles_init<true, ELTWISE_OP_TYPE>(cb_in0, cb_in1);

    // =============================================================================
    // MAIN PROCESSING LOOP - PROCESS SINGLE TILES
    // =============================================================================
    // Process one tile at a time through the entire pipeline:
    // Binary Op Init → HW Startup → CB Sync → Eltwise Op → CB Sync → Reduce Init → Reduce Op → Release CB

    uint32_t total_tiles = 1;

    for (uint32_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        // =========================================================================
        // STEP 1: BINARY OP INIT + HW STARTUP FOR BINARY OP
        // =========================================================================

        // =========================================================================
        // STEP 2: CB SYNC (Wait for input tiles)
        // =========================================================================
        cb_wait_front(cb_inp0, 1);            // Wait for first input tile
        cb_wait_front(cb_inp1, 1);            // Wait for second input tile
        cb_reserve_back(cb_intermediate, 1);  // Reserve space for output tile

        // =========================================================================
        // STEP 3: ELTWISE OP
        // =========================================================================
        tile_regs_acquire();
        // *** ACTUAL ELTWISE OPERATION ***
        ELTWISE_OP(cb_inp0, cb_inp1, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_intermediate);  // Pack result to intermediate CB
        tile_regs_release();

        // =========================================================================
        // STEP 4: CB SYNC (Clean up eltwise inputs, make intermediate available)
        // =========================================================================
        cb_pop_front(cb_inp0, 1);          // Done with first input
        cb_pop_front(cb_inp1, 1);          // Done with second input
        cb_push_back(cb_intermediate, 1);  // Make eltwise result available

        // =========================================================================
        // STEP 5: REDUCE INIT (only on first tile)
        // =========================================================================

        // compute_kernel_hw_startup(cb_intermediate, cb_in2, cb_out0);
        reduce_init(cb_intermediate, cb_in2, cb_out0);  // cb_in2 is scaler
        cb_wait_front(cb_in2, 1);                       // Wait for scaler tile

        // =========================================================================
        // STEP 6: REDUCE OP
        // =========================================================================
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;

        acquire_dst();  // Get exclusive access to destination registers

        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // Wait for the intermediate tile (eltwise result)
            cb_wait_front(cb_intermediate, onetile);

            // *** ACTUAL REDUCE OPERATION ***
            // REDUCE_OP is expected to come from add_define (typically ADD for sum reduction)
            reduce_tile(cb_intermediate, cb_in2, 0, 0, reduce_dst_idx);

            cb_pop_front(cb_intermediate, onetile);  // Done with intermediate tile
        }

        // Debug: Print the destination register contents
        dprint_tensix_dest_reg(reduce_dst_idx);

        // Pack and output the reduced result
        cb_reserve_back(cb_out0, onetile);
        pack_tile(reduce_dst_idx, cb_out0);
        cb_push_back(cb_out0, onetile);

        release_dst();  // Release destination registers
    }

    // =============================================================================
    // CLEANUP PHASE
    // =============================================================================
    reduce_uninit();  // Clean up reduce operation resources
}
}  // namespace NAMESPACE
