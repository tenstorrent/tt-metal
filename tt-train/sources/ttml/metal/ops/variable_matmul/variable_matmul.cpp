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
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t in0_row_offset_tiles,
    uint32_t effective_M_tiles) {
    return ttnn::prim::ttml_variable_matmul(
        input_tensor, weight_tensor, config, compute_kernel_config, in0_row_offset_tiles, effective_M_tiles);
}

}  // namespace ttml::metal
