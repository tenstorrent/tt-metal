// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
void smoothstep_tile_face() {
    // SFPU Tile Organization:
    // Each tile in Dst registers is divided into four 16x16 faces.
    // n_vector_in_face = 32 as there are 32 SIMD lanes per tile
    float edge0 = 0;
    float edge1 = 1.0;
    constexpr uint32_t n_vector_in_face = 32;

    // Calculate base indices for each tile in the Dst register array.
    const uint32_t in0_base_idx = 0;

    // Process one face of the tile (8 SIMD operations).
    // Implements smoothstep: result = t * t * (3 - 2 * t), where t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
    float inv_delta = 1.0f / (edge1 - edge0);
    for (size_t i = 0; i < 8; i++) {
        vFloat x = dst_reg[in0_base_idx + i];
        vFloat t = (x - edge0) * inv_delta;
        // t = min(max(t, 0.0f), 1.0f);
        v_if(t < 0.0f) { t = 0.0f; }
        v_elseif(t > 1.0f) { t = 1.0f; }
        v_endif;
        vFloat result = t * t * (3.0f - 2.0f * t);
        dst_reg[in0_base_idx + i] = result;
    }
    // Note: This function only processes ONE FACE of a tile.
    // The _llk_math_eltwise_binary_sfpu_params_ wrapper (used in my_smoothstep_tile_internal)
    // will call this function for each face in the tile (typically 4 times for a 32x32 tile).
}

/**
 * SFPU Kernel Wrapper Function
 *
 * This function bridges between the high-level kernel API and the low-level SFPU function.
 * It demonstrates the typical SFPU programming pattern:
 *
 * 1. The custom SFPU function (add_tile_face) operates on individual tile faces
 * 2. The LLK (Low-Level Kernel) wrapper handles the face iteration automatically
 * 3. The _llk_math_eltwise_binary_sfpu_params_ template:
 *    - Takes our custom function as a parameter
 *    - Calls it for each face in the tiles (typically 4 times per tile)
 *
 * Template parameter <false> indicates this is a non-approximated operation. Which might
 * be consumed by downstream functions to trade off accuracy for performance.
 *
 * This function is only compiled for the MATH core (TRISC_MATH context).
 */
inline void my_smoothstep_tile_internal(uint32_t idx_dst0, float edge0, float edge1) {
    // LLK wrapper that calls smoothstep_tile_face for each face in the tiles.
    // Parameters: (custom_function, input_dst_idx, output_dst_idx, edge0, edge1)
    _llk_math_eltwise_unary_sfpu_params_<false>(smoothstep_tile_face, idx_dst0);
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
 * 2. my_add_tile_internal()  - LLK wrapper (invokes add_tile_face for each face)
 * 3. my_add_tiles()          - High-level API (easier to use in kernel code)
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
inline void my_smoothstep_tiles(uint32_t idx_dst0, float edge0, float edge1) {
    MATH(my_smoothstep_tile_internal(idx_dst0, edge0, edge1));
}

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    // and write to the output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Tell the SFPU that we will be using circular buffers c_in0 and c_out0
    // to perform the computation.
    init_sfpu(cb_in0, cb_out0);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in the input circular buffer
        cb_wait_front(cb_in0, 1);
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        tile_regs_acquire();
        // Copy the tiles from the circular buffers into Dst registers
        copy_tile(cb_in0, 0, 0);           // input x
        my_smoothstep_tiles(0, 0.0, 1.0);  // <-- The custom SFPU smoothstep happens here
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
        // Release the held register
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
