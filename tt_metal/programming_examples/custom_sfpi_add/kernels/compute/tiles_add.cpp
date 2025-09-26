// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

// The SFPU itself only available on the MATH core. The TRISC_MATH macro
// is defined when the code is being compiled for the MATH core.
#ifdef TRISC_MATH

/**
 * The SFPU (Special Function Processing Unit) is the vector engine in Tensix that operates
 * on data already loaded into the Dst registers. Unlike the matrix engine (FPU) which takes
 * circular buffer indices as parameters, SFPU functions work with Dst register indices.
 *
 * Key SFPU Concepts:
 * - SFPU operates on 32x32 tiles organized as tile faces
 * - Each tile face contains 32 SIMD lanes of data
 * - A full tile has multiple faces (typically 4 faces for a 32x32 tile)
 * - SFPU processes one face at a time in a loop
 *
 * @param dst_index_in0  Index of the first input tile in Dst registers (not CB index!)
 * @param dst_index_in1  Index of the second input tile in Dst registers (not CB index!)
 * @param dst_index_out  Index of the output tile in Dst registers (not CB index!)
 */
inline void my_add_tile_face(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    // SFPU Tile Organization:
    // Each tile in Dst registers is divided into four 16x16 faces.
    // n_vector_in_tile = 32 as there are 32 SIMD lanes per tile
    // i.e.
    //   dst_reg[0:31] holds the first tile
    //          dst_reg[0:8] holds the first face of the first tile
    //          dst_reg[8:15] holds the second face of the first tile,
    //   dst_reg[32:63] holds the second tile
    //          dst_reg[32:40] holds the first face of the second tile
    //          dst_reg[40:47] holds the second face of the second tile,
    //         etc.
    // NOTE: This value is architectural dependent and may change in future hardware revisions.
    //       For Blackhole and Whitehole, SFPU is 32 elements wide.
    constexpr uint32_t n_vector_in_tile = 32;

    // Calculate base indices for each tile in the Dst register array.
    // Each tile occupies multiple consecutive slots in dst_reg[].
    // The multiplication accounts for the face structure within each tile.
    const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

    // Process one face of the tile (8 SIMD operations).
    // Why 8 iterations? Each iteration processes 32 elements (vFloat is 32-elements-wide SIMD),
    // so 32 * 8 = 16*16 (a full face).
    //
    // SFPU Programming Pattern:
    // 1. Load data from Dst registers into vFloat SIMD variables
    // 2. Perform SIMD arithmetic operations
    // 3. Store results back to Dst registers
    for (size_t i = 0; i < 8; i++) {
        // Load 32-element SIMD vectors from Dst registers.
        // vFloat represents 32 parallel floating-point values.
        // This is the SFPU's native SIMD data type.
        vFloat a = dst_reg[in0_base_idx + i];
        vFloat b = dst_reg[in1_base_idx + i];

        // Perform SIMD addition: all 32 elements are added in parallel.
        // This is where the actual computation happens on the vector engine.
        // For FP32 accuracy, ensure the host sets fp32_dest_acc_en=true.
        dst_reg[out_base_idx + i] = a + b;

        // The above program can be shortened to a single line:
        // However, the expanded form is clearer for educational purposes.
        // dst_reg[out_base_idx] = dst_reg[in0_base_idx] + dst_reg[in1_base_idx];
    }

    // Note: This function only processes ONE FACE of a tile.
    // The _llk_math_eltwise_binary_sfpu_params_ wrapper  will call this function
    // for each face in the tile (typically 4 times for a 32x32 tile).
}
#endif

/**
 * High-Level SFPU API Function
 *
 * This is the public interface that kernel code calls to perform custom SFPU operations.
 * It demonstrates the multi-layer abstraction in TT-Metal kernel programming:
 *
 * Abstraction Layers (from low to high):
 * 1. add_tile_face()         - Low-level SFPU function (operates on tile faces)
 * 2. my_add_tile()          - High-level API (easier to use in kernel code)
 *
 * The MATH() macro is crucial - it ensures this function only executes on cores
 * that is designated to the SFPU (vector engine). This is part of TT-Metal's
 * heterogeneous compute model where different cores have different capabilities.
 *
 * Usage Pattern:
 * - Call this function after loading data into Dst registers with copy_tile()
 * - Specify Dst register indices (not circular buffer indices!)
 * - Results are written back to the specified Dst register
 *
 * This function takes tile idx_dst0 and idx_dst1 as inputs, adds them,
 * and writes the result to tile idx_out0 in the Dst registers.
 */
inline void my_add_tile(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(my_add_tile_face, idx_dst0, idx_dst1, idx_out0));
}

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    // and write to the output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Tell the SFPU that we will be using circular buffers c_in0 and c_out0
    // to perform the computation.
    init_sfpu(cb_in0, cb_out0);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in both input circular buffers
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        tile_regs_acquire();
        // Copy the tiles from the circular buffers into Dst registers
        copy_tile(cb_in0, 0, 0);
        copy_tile(cb_in1, 0, 1);
        my_add_tile(0, 1, 0);  // <-- The custom SFPU addition happens here
        // Finished the computation, transfer register ownership to the unpacker
        tile_regs_commit();
        tile_regs_wait();
        // Make sure there is space in the output circular buffer
        cb_reserve_back(cb_out0, 1);
        // Copy the result from adding the tiles to the output circular buffer
        pack_tile(0, cb_out0);
        // Mark the output tile as ready and pop the input tiles
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        // Release the held register
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
