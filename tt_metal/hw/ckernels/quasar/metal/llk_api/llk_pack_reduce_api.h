// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"

/*************************************************************************
 * LLK PACK REDUCE MASK CONFIGURATION
 *************************************************************************/

/**
 * @brief Configures PACKER0 edge mask programming to support reduce operations (Quasar native API).
 *
 * ocb is unused on Quasar (different packer architecture); accepted to keep
 * the API arch-agnostic with Blackhole/Wormhole B0 callers that thread the output CB through.
 *
 * @tparam reduce_dim: The reduce op dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @tparam pack_mode: Unused on Quasar
 * @param ocb: The output Dataflow Buffer identifier
 */
template <ReduceDim reduce_dim, [[maybe_unused]] PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(std::uint32_t ocb) {
    static_assert(pack_mode == PackMode::Default, "Quasar pack reduce mask does not support pack_mode != Default");
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(ocb);
    _llk_pack_reduce_mask_config_<reduce_dim>(tensor_shape);
}

/**
 * @brief Clears PACKER0 edge mask configuration to restore normal packing behavior after reduce operations
 */
inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
