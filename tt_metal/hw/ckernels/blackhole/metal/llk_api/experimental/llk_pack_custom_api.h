// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_io.h"
#include "llk_outputs.h"
#include "experimental/llk_pack_custom.h"

/*************************************************************************
 * LLK PACK CUSTOM API - Lightweight MOP outer-loop patching
 *************************************************************************/

// WARNING: Experimental API for SDPA optimizations only.

// Lightweight MOP outer-loop patch: only updates mop_cfg[0] (= num_faces * num_tiles).
// Use ONLY after an initial full llk_pack_mop_config has programmed all 9 MOP registers.
// Safe when all CBs share the same tile format (same num_faces, face_r_dim, tile_c_dim).
inline void llk_pack_set_mop_outer_loop(const std::uint32_t output, std::uint32_t num_tiles) {
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    _llk_pack_set_mop_outer_loop_(num_faces, num_tiles);
}
