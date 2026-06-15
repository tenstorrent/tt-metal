// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variable_matmul.hpp"

#include "device/variable_matmul_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor variable_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const VariableMatmulConfig& config,
    const ttnn::Tensor& offsets_tensor,
    OffsetsRole offsets_role,
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::Tensor> output_tensor,
    uint32_t offsets_start_index,
    uint32_t expected_M_tiles) {
    return ttnn::prim::ttml_variable_matmul(
        input_tensor,
        weight_tensor,
        config,
        offsets_tensor,
        offsets_role,
        transpose_a,
        transpose_b,
        compute_kernel_config,
        output_tensor,
        offsets_start_index,
        expected_M_tiles);
}

}  // namespace ttml::metal
