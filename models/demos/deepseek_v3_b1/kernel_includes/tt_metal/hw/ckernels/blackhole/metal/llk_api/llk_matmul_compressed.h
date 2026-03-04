// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_compressed.h"

// NOTE: Requires matmul.h to be included before this header.
// Depends on MM_THROTTLE, llk_math_matmul, _llk_unpack_AB_matmul_.

namespace compressed {

// ---------------------------------------------------------------------------
// Per-tile matmul with compressed in1 (weights)
//
// In matmul: in0 (activations) → srcB, in1 (weights) → srcA (swapped!)
// So compressed weights need reconfig_unpack_srca, not srcb.
// ---------------------------------------------------------------------------

/**
 * @brief Matmul one A tile with one compressed B tile.
 *
 * Reads the format from the assignment array, skips bfp0 tiles,
 * reconfigures srcA unpacker, then does unpack + math.
 * Returns the compressed tile size in shifted units (0 for bfp0).
 */
FORCE_INLINE uint32_t matmul_tiles_in1_compressed(
    const volatile uint8_t* assign_ptr,
    uint32_t tile_idx,
    uint32_t addr_a,
    uint32_t addr_b,
    uint32_t tile_index_a,
    uint32_t tile_size_a,
    bool partial_face_a,
    bool partial_face_b,
    uint32_t dst_index) {
    uint32_t fmt = get_tile_format(assign_ptr, tile_idx);
    uint32_t tile_size_b = TILE_SIZES[fmt] >> cb_addr_shift;

    if (fmt != FMT_BFP0) {
        reconfig_unpack_srca(fmt);
        UNPACK((_llk_unpack_AB_matmul_(
            addr_a,
            addr_b,
            tile_index_a,
            0,  // tile_index_b: addr_b already points to the tile
            tile_size_a,
            tile_size_b,
            partial_face_a,
            partial_face_b)));
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(dst_index)));
    }

    return tile_size_b;
}

}  // namespace compressed
