// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_copy_dest_values.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_init.h"

namespace ckernel {

// Generalized version that takes a DataFormat template parameter
template <DataFormat DATA_FORMAT>
void llk_math_eltwise_binary_sfpu_copy_dest_values(
    uint32_t dst_index_in, uint32_t dst_index_out, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::copy_dest_value<DATA_FORMAT, APPROXIMATE>, dst_index_in, dst_index_out, 0 /*unused*/, vector_mode);
}

// Deprecated: Use the template version with DataFormat parameter instead
[[deprecated("Use llk_math_eltwise_binary_sfpu_copy_dest_values<DataFormat> instead")]]
void llk_math_eltwise_binary_sfpu_copy_dest_values(
    uint32_t dst_index_in, uint32_t dst_index_out, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::copy_dest_value<APPROXIMATE>, dst_index_in, dst_index_out, 0 /*unused*/, vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_copy_dest_values_init() {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::copy_dest_value_init);
}

}  // namespace ckernel
