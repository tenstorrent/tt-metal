// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_matmul.hpp"

#include "device/gram_matmul_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input_tensor,
    const std::optional<const ttml::metal::ops::gram_matmul::device::GramMatmulConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& output) {
    return ttnn::prim::ttml_gram_matmul(input_tensor, config, memory_config, dtype, compute_kernel_config, output);
}

}  // namespace ttml::metal
