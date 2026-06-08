// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

/**
 * @brief Configure the packer edge-offset masks and tile-row-set mapping for a reduce output.
 *
 * Uses the runtime face_r_dim of the output CB so that only the reduced datums survive; required for
 * narrow tiles (e.g. tile_dimensions=[1,32]) where face_r_dim differs from FACE_R_DIM.
 *
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @param ocb: Circular-buffer index of the reduce output.
 * @note Pairs with @ref llk_math_reduce on the math thread, whose reduced output these masks gate.
 * @note Call @ref llk_pack_reduce_mask_clear to restore the default pass-through masks.
 */
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(uint32_t ocb) {
    const std::uint32_t output_id = get_output_id(ocb);
    _llk_pack_reduce_mask_config_<dim, pack_mode>(get_output_face_r_dim(output_id));
}

/**
 * @brief Restore the default packer edge masks and tile-row-set mapping after a reduce.
 *
 * @note Pairs with @ref llk_pack_reduce_mask_config.
 */
inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
