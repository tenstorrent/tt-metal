// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_param_structs.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

template <bool untilize = false, ReduceDim dim>
inline void llk_pack_reduce_mask_config() {
    _llk_pack_reduce_mask_config_<untilize, dim>();
}

inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
