// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

// Pass the output geometry through one shared LLK mask configuration path.
template <PoolType reduce_type, ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config_impl(
    const std::uint32_t pack_dst_format, const ckernel::TensorShape& tensor_shape) {
    _llk_pack_reduce_mask_config_<reduce_type, dim, pack_mode>(pack_dst_format, tensor_shape);
}

// Get the output format and shape from the output CB.
template <PoolType reduce_type, ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(uint32_t ocb) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(ocb);
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);
    llk_pack_reduce_mask_config_impl<reduce_type, dim, pack_mode>(get_output_dst_format(output_id), tensor_shape);
}

inline void llk_pack_reduce_mask_clear() {
    SAN_HOOK(unsupported());
    _llk_pack_reduce_mask_clear_();
}
