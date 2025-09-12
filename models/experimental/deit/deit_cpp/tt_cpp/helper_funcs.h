#ifndef DEIT_CPP_TT_CPP_HELPER_FUNCS_H
#define DEIT_CPP_TT_CPP_HELPER_FUNCS_H

#include "deit_config.h"
#include <optional>
#include <functional>
#include <torch/torch.h>

namespace helper_funcs {

/**
 * Linear transformation function
 * Performs weight * input + bias operation
 * 
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Optional bias tensor
 * @param output_mem_config Memory configuration for output tensor
 * @return Output tensor after linear transformation
 */
ttnn::Tensor linear_transform(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    std::optional<ttnn::Tensor> bias,
    ttnn::MemoryConfig output_mem_config = ttnn::DRAM_MEMORY_CONFIG
);

/**
 * Convert torch::Tensor to ttnn::Tensor with TILE layout
 * Ensures tensor has at least 4 dimensions and uses TILE layout
 * 
 * @param tensor Input torch tensor
 * @param device Target mesh device
 * @param shape Optional target shape (defaults to tensor's shape with padding to 4D)
 * @return ttnn::Tensor converted from torch tensor with TILE layout
 */
ttnn::Tensor torch_to_tt_tensor_tile(
    const at::Tensor& tensor,
    ttnn::MeshDevice* device,
    std::optional<std::vector<int64_t>> shape = std::nullopt
);

/**
 * Convert torch::Tensor to ttnn::Tensor
 * 
 * @param tensor Input torch tensor
 * @param dtype Optional target data type
 * @param layout Optional target layout
 * @return Converted ttnn tensor
 */
ttnn::Tensor from_torch(
    const at::Tensor& tensor,
    std::optional<ttnn::DataType> dtype = std::nullopt,
    std::optional<ttnn::Layout> layout = std::nullopt
);

} // namespace helper_funcs

#endif // DEIT_CPP_TT_CPP_HELPER_FUNCS_H