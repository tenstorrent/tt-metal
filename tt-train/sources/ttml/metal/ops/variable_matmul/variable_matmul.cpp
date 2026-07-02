// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "variable_matmul.hpp"

#include "device/variable_matmul_device_operation.hpp"

namespace ttml::metal {

using ops::variable_matmul::device::OffsetsRole;

ttnn::Tensor variable_matmul_into_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    const ttnn::Tensor& output_tensor,
    uint32_t offsets_start_index,
    uint32_t expected_M_tiles,
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::ttml_variable_matmul(
        input_tensor,
        weight_tensor,
        config,
        offsets_tensor,
        OffsetsRole::InputAndOutputRow,
        transpose_a,
        transpose_b,
        compute_kernel_config,
        output_tensor,
        offsets_start_index,
        expected_M_tiles);
}

ttnn::Tensor variable_matmul_k_sliced(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    uint32_t offsets_start_index,
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::ttml_variable_matmul(
        input_tensor,
        weight_tensor,
        config,
        offsets_tensor,
        OffsetsRole::InputAndWeightK,
        transpose_a,
        transpose_b,
        compute_kernel_config,
        /*output_tensor=*/std::nullopt,
        offsets_start_index,
        /*expected_M_tiles=*/0);
}

}  // namespace ttml::metal
