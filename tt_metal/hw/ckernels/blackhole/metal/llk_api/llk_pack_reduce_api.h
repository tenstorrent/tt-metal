// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

// Use the runtime face_r_dim of the output CB. Required for narrow tiles
// (e.g. tile_dimensions=[1,32]) where face_r_dim differs from FACE_R_DIM.
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(uint32_t ocb) {
    const std::uint32_t output_id = get_output_id(ocb);
    _llk_pack_reduce_mask_config_<dim, pack_mode>(get_output_face_r_dim(output_id));
}

inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
