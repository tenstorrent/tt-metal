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
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t in0_row_offset_tiles,
    uint32_t effective_M_tiles,
    uint32_t in0_k_offset_tiles,
    uint32_t in1_k_offset_tiles,
    std::optional<ttnn::Tensor> output_tensor,
    uint32_t out_row_offset_tiles,
    std::optional<ttnn::Tensor> offsets_tensor,
    OffsetsRole offsets_role,
    uint32_t offsets_start_index) {
    return ttnn::prim::ttml_variable_matmul(
        input_tensor,
        weight_tensor,
        config,
        transpose_a,
        transpose_b,
        compute_kernel_config,
        in0_row_offset_tiles,
        effective_M_tiles,
        in0_k_offset_tiles,
        in1_k_offset_tiles,
        output_tensor,
        out_row_offset_tiles,
        offsets_tensor,
        offsets_role,
        offsets_start_index);
}

}  // namespace ttml::metal
