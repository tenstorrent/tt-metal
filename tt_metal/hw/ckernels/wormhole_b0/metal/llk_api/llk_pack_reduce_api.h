// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

// Pass the output geometry through one shared LLK mask configuration path.
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config_impl(const ckernel::TensorShape& tensor_shape) {
    _llk_pack_reduce_mask_config_<dim, pack_mode>(tensor_shape);
}

// Get the output shape from the output CB.
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(std::uint32_t ocb) {
    llk_pack_reduce_mask_config_impl<dim, pack_mode>(get_output_tensor_shape(get_output_id(ocb)));
}

inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
