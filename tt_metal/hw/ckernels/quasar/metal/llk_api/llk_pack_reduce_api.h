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
 * @tparam untilize Unused on Quasar
 * @tparam reduce_dim The reduce op dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @tparam pack_mode Unused on Quasar
 */
template <ReduceDim reduce_dim, [[maybe_unused]] PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config() {
    static_assert(pack_mode == PackMode::Default, "Quasar pack reduce mask does not support pack_mode != Default");
    _llk_pack_reduce_mask_config_<reduce_dim>();
}

/**
 * @brief Clears PACKER0 edge mask configuration to restore normal packing behavior after reduce operations
 */
inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
