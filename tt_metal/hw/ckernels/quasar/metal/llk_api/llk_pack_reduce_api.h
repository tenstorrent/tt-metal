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
 * @tparam reduce_dim The reduce op dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 */
template <ReduceDim reduce_dim>
inline void llk_pack_reduce_mask_config() {
    _llk_pack_reduce_mask_config_<reduce_dim>();
}

/**
 * @brief WH/BH-style two-template-parameter form; untilize must be false on Quasar.
 */
template <bool untilize, ReduceDim reduce_dim>
inline void llk_pack_reduce_mask_config() {
    static_assert(!untilize, "Quasar pack reduce mask does not support untilize=true");
    _llk_pack_reduce_mask_config_<reduce_dim>();
}

/**
 * @brief Clears PACKER0 edge mask configuration to restore normal packing behavior after reduce operations
 */
inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
