#ifndef DEIT_CPP_TT_CPP_HELPER_FUNCS_H
#define DEIT_CPP_TT_CPP_HELPER_FUNCS_H

#include "tt_cpp/deit_config.h"
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
    std::shared_ptr<ttnn::MeshDevice> device,
    std::optional<std::vector<int64_t>> shape = std::nullopt
);

/**
 * Template function to create concrete tensor from torch tensor and tensor spec
 * 
 * @param contiguous_tensor Input torch tensor (must be contiguous)
 * @param spec Tensor specification for the output tensor
 * @return ttnn::Tensor created from the input data
 */
template<typename T>
ttnn::Tensor create_concrete(torch::Tensor &contiguous_tensor, tt::tt_metal::TensorSpec &spec);

/**
 * Template function to create row-major host buffer
 * 
 * @param host_buffer Input host buffer
 * @param tensor_spec Tensor specification
 * @param padded_output Whether to return padded output
 * @return Row-major host buffer
 */
template <typename T>
tt::tt_metal::HostBuffer create_row_major_host_buffer(
    tt::tt_metal::HostBuffer host_buffer, 
    const tt::tt_metal::TensorSpec& tensor_spec, 
    const bool padded_output
);

/**
 * Get host buffer from tensor
 * 
 * @param tt_tensor Input ttnn tensor (must be on host)
 * @param padded_output Whether to return padded output
 * @return Host buffer extracted from tensor
 */
tt::tt_metal::HostBuffer get_host_buffer_from_tensor(
    const ttnn::Tensor& tt_tensor, 
    const bool padded_output = false
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
    std::optional<ttnn::Layout> layout = std::nullopt);

/**
 * Convert ttnn::Tensor to torch::Tensor
 * 
 * @param tensor Input ttnn tensor
 * @param padded_output Whether to return padded output
 * @return Converted torch tensor
 */
torch::Tensor to_torch(const ttnn::Tensor& tensor, const bool padded_output = false);

/**
 * Apply layer normalization
 * 
 * @param input Input tensor
 * @param weight Weight tensor for normalization
 * @param bias Bias tensor for normalization
 * @param eps Epsilon value for numerical stability
 * @return Normalized tensor
 */
ttnn::Tensor apply_layernorm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias,
    float eps = 1e-5
);

/**
 * Load and preprocess an image for DeiT inference
 * Mimics the functionality of AutoImageProcessor from transformers
 * @param image_path Path to the image file
 * @return Preprocessed image tensor [1, 3, 224, 224] ready for DeiT inference
 */
torch::Tensor load_and_preprocess_image(
    const std::string& image_path
);

/**
 * Compute Pearson Correlation Coefficient (PCC) between two tensors
 * @param tensor1 First tensor
 * @param tensor2 Second tensor
 * @return PCC value
 */
double compute_pcc(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

} // namespace helper_funcs

#endif // DEIT_CPP_TT_CPP_HELPER_FUNCS_H