// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config() {
    _llk_pack_reduce_mask_config_<dim, pack_mode>();
}

inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
