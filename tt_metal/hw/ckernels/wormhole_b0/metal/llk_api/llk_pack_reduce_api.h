// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

// Select the LLK specialization locally so callers can keep passing a runtime output CB.
template <PoolType reduce_type, ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config_impl(const std::uint32_t face_r_dim, const TileGeometry geometry) {
    switch (geometry) {
        case TileGeometry::Faces1x1:
            _llk_pack_reduce_mask_config_<reduce_type, dim, pack_mode, TileGeometry::Faces1x1>(face_r_dim);
            break;
        case TileGeometry::Faces1x2:
            _llk_pack_reduce_mask_config_<reduce_type, dim, pack_mode, TileGeometry::Faces1x2>(face_r_dim);
            break;
        case TileGeometry::Faces2x1:
            _llk_pack_reduce_mask_config_<reduce_type, dim, pack_mode, TileGeometry::Faces2x1>(face_r_dim);
            break;
        case TileGeometry::Faces2x2:
            _llk_pack_reduce_mask_config_<reduce_type, dim, pack_mode, TileGeometry::Faces2x2>(face_r_dim);
            break;
    }
}

// Derive the face grid from the output CB's tile dimensions and face height.
template <PoolType reduce_type, ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(uint32_t ocb) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(ocb);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const auto geometry =
        get_tile_geometry(get_output_tile_r_dim(output_id) / face_r_dim, get_output_tile_c_dim(output_id) / FACE_C_DIM);
    llk_pack_reduce_mask_config_impl<reduce_type, dim, pack_mode>(face_r_dim, geometry);
}

inline void llk_pack_reduce_mask_clear() {
    SAN_HOOK(unsupported());
    _llk_pack_reduce_mask_clear_();
}
