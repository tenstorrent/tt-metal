// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "helper_funcs.h"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/distributed.hpp>
#include <torch/torch.h>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace helper_funcs {

ttnn::Tensor linear_transform(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    std::optional<ttnn::Tensor> bias,
    ttnn::MemoryConfig output_mem_config
) {
    // Transpose weight for matrix multiplication
    auto weight_transposed = ttnn::transpose(weight, -2, -1);
    
    // // Perform matrix multiplication
    // auto output = ttnn::matmul(input, weight_transposed);
    
    // Add bias if provided
    if (bias.has_value() && bias->layout() != ttnn::TILE_LAYOUT) {
        bias = ttnn::to_layout(bias.value(), ttnn::TILE_LAYOUT, std::nullopt, std::nullopt);
    }
    
    // Perform linear transformation
    auto output = ttnn::linear(input, weight_transposed, bias, false, false, output_mem_config);
    
    return output;
}

ttnn::Tensor torch_to_tt_tensor_tile(
    const at::Tensor& tensor,
    std::shared_ptr<ttnn::MeshDevice> device,
    std::optional<std::vector<int64_t>> shape
) {
    // Get tensor shape, default to tensor's current shape if not provided
    std::vector<int64_t> target_shape;
    if (shape.has_value()) {
        target_shape = shape.value();
    } else {
        // Convert at::IntArrayRef to std::vector<int64_t>
        auto tensor_sizes = tensor.sizes();
        target_shape = std::vector<int64_t>(tensor_sizes.begin(), tensor_sizes.end());
        
        // Ensure at least 4 dimensions by padding with 1s at the beginning
        while (target_shape.size() < 4) {
            target_shape.insert(target_shape.begin(), 1);
        }
    }
    
    // Reshape tensor to target shape
    at::Tensor reshaped_tensor = tensor.reshape(target_shape);
    
    // Ensure tensor is contiguous
    at::Tensor contiguous_tensor = reshaped_tensor.contiguous();
    
    // Create logical shape for ttnn
    ttnn::Shape logical_shape(std::vector<uint32_t>{target_shape.begin(), target_shape.end()});
    
    ttnn::Tensor tt_tensor = from_torch(contiguous_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    tt_tensor = tt_tensor.to_device(device.get());
    
    // Move tensor to device - use the same approach as from_torch function
    return tt_tensor;
}

template<typename T>
ttnn::Tensor create_concrete(torch::Tensor &contiguous_tensor, tt::tt_metal::TensorSpec &spec)
{
    return ttnn::Tensor::template from_span<T>(
            tt::stl::template Span<const T>(reinterpret_cast<T*>(contiguous_tensor.data_ptr()), contiguous_tensor.numel()),
            spec);
}


template <typename T>
tt::tt_metal::HostBuffer create_row_major_host_buffer(
    tt::tt_metal::HostBuffer host_buffer, const tt::tt_metal::TensorSpec& tensor_spec, const bool padded_output) {
    if (padded_output) {
        if (tensor_spec.layout() == ttnn::Layout::TILE) {
            auto row_major_data = tt::tt_metal::tensor_impl::convert_layout_tile_to_row_major(
                tensor_spec.physical_shape(), tensor_spec.tile(), host_buffer.view_as<const T>());
            return tt::tt_metal::HostBuffer(std::move(row_major_data));
        }
        return host_buffer;
    }

    // No modifications needed; direclty return buffer
    if (tt::tt_metal::tensor_impl::logical_matches_physical(tensor_spec)) {
        return host_buffer;
    }

    auto logical_data = tt::tt_metal::tensor_impl::decode_tensor_data(host_buffer.view_as<const T>(), tensor_spec);
    return tt::tt_metal::HostBuffer(std::move(logical_data));
}


tt::tt_metal::HostBuffer get_host_buffer_from_tensor(const ttnn::Tensor& tt_tensor, const bool padded_output) {
    TT_ASSERT(tt::tt_metal::is_cpu_tensor(tt_tensor), "Tensor must be on host for padding");

    const auto& tensor_spec = tt_tensor.tensor_spec();
    auto convert_to_logical = [&tensor_spec, padded_output](const tt::tt_metal::HostBuffer& buffer) {
        const auto tt_dtype = tensor_spec.data_type();
        switch (tt_dtype) {
            case ttnn::DataType::UINT8: {
                return create_row_major_host_buffer<uint8_t>(buffer, tensor_spec, padded_output);
            }
            case ttnn::DataType::UINT16: {
                return create_row_major_host_buffer<uint16_t>(buffer, tensor_spec, padded_output);
            }
            case ttnn::DataType::INT32: {
                return create_row_major_host_buffer<int32_t>(buffer, tensor_spec, padded_output);
            }
            case ttnn::DataType::UINT32: {
                return create_row_major_host_buffer<uint32_t>(buffer, tensor_spec, padded_output);
            }
            case ttnn::DataType::FLOAT32: {
                return create_row_major_host_buffer<float>(buffer, tensor_spec, padded_output);
            }
            case ttnn::DataType::BFLOAT16: {
                return create_row_major_host_buffer<bfloat16>(buffer, tensor_spec, padded_output);
            }
            case ttnn::DataType::BFLOAT8_B:
            case ttnn::DataType::BFLOAT4_B: {
                const auto& tile = tensor_spec.tile();
                tt::stl::Span<const std::uint32_t> uint32_data = tt::tt_metal::host_buffer::get_as<std::uint32_t>(buffer);
                auto float_unpacked_data = tt_dtype == ttnn::DataType::BFLOAT8_B
                                               ? unpack_bfp8_tiles_into_float_vec(
                                                     uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                                               : unpack_bfp4_tiles_into_float_vec(
                                                     uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
                auto input_float_buffer = tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
                return create_row_major_host_buffer<float>(input_float_buffer, tensor_spec, padded_output);
            }
            default: {
                TT_THROW("Unsupported DataType: {}", static_cast<int>(tt_dtype));
                break;
            }
        }
    };

    return convert_to_logical(std::visit(
        tt::stl::overloaded{
            [](const tt::tt_metal::HostStorage& storage) {
                std::vector<tt::tt_metal::HostBuffer> buffers;
                storage.buffer().apply([&buffers](const tt::tt_metal::HostBuffer& shard) { buffers.push_back(shard); });
                TT_FATAL(
                    buffers.size() == 1,
                    "Can't convert a tensor distributed on {} mesh to row-major logical tensor. Supply a mesh composer "
                    "to concatenate multi-device shards.",
                    storage.buffer().shape());
                return buffers.front();
            },
            [&tt_tensor](auto&&) -> tt::tt_metal::HostBuffer {
                TT_THROW(
                    "Tensor with {} cannot be converted to torch",
                    tt::stl::get_active_type_name_in_variant(tt_tensor.storage()));
            },
        },
        tt_tensor.storage()));
}

ttnn::Tensor from_torch(const at::Tensor& tensor,
    std::optional<ttnn::DataType> dtype,    
    std::optional<ttnn::Layout> layout) {
    auto torch_dtype = tensor.scalar_type();
    auto torch_shape = tensor.sizes();
    ttnn::Shape logical_shape(std::vector<uint32_t>{torch_shape.begin(), torch_shape.end()});
    ttnn::DataType data_type;
    if (dtype.has_value()) {
        data_type = dtype.value();
    } else if (torch_dtype == torch::kFloat) {
        data_type = ttnn::DataType::FLOAT32;
    } else if (torch_dtype == torch::kHalf) {
        data_type = ttnn::DataType::BFLOAT16;
    } else if (torch_dtype == torch::kBFloat16) {
        data_type = ttnn::DataType::BFLOAT16;
    } else if (torch_dtype == torch::kInt64) {
        // TODO: add DataType::INT64?
        data_type = ttnn::DataType::UINT32;
    } else if (torch_dtype == torch::kInt32) {
        data_type = ttnn::DataType::INT32;
    } else if (torch_dtype == torch::kInt16) {
        // TODO: add DataType::INT16?
        data_type = ttnn::DataType::UINT16;
    } else if (torch_dtype == torch::kUInt8) {
        data_type = ttnn::DataType::UINT8;
    } else {
        TT_THROW("from_torch Unsupported type : {}", c10::toString(torch_dtype));
    }

    if (data_type == ttnn::DataType::BFLOAT8_B || data_type == ttnn::DataType::BFLOAT4_B) {
        throw std::runtime_error("from_torch: bfloat8_b/bfloat4_b unsupported!");
    }
    
    torch::Tensor contiguous_tensor = tensor.contiguous();
    auto maybe_convert_tensor_dtype = [&torch_dtype, &contiguous_tensor](c10::ScalarType target_py_dtype) {
        if (torch_dtype != target_py_dtype) {
            contiguous_tensor = contiguous_tensor.to(target_py_dtype);
        }
    };

    auto tensor_spec = tt::tt_metal::TensorSpec(logical_shape, 
                                tt::tt_metal::TensorLayout(data_type, tt::tt_metal::PageConfig(layout.value_or(ttnn::Layout::ROW_MAJOR)),
                                                            ttnn::MemoryConfig{})
                                );
    switch (data_type) {
        case ttnn::DataType::UINT8: {
            maybe_convert_tensor_dtype(torch::kUInt8);
            return create_concrete<uint8_t>(contiguous_tensor, tensor_spec);
        }
        case ttnn::DataType::UINT16: {
            maybe_convert_tensor_dtype(torch::kUInt16);
            return create_concrete<uint16_t>(contiguous_tensor, tensor_spec);
        }
        case ttnn::DataType::INT32: {
            maybe_convert_tensor_dtype(torch::kInt32);
            return create_concrete<int32_t>(contiguous_tensor, tensor_spec);
        }
        case ttnn::DataType::UINT32: {
            maybe_convert_tensor_dtype(torch::kUInt32);
            return create_concrete<uint32_t>(contiguous_tensor, tensor_spec);
        }
        case ttnn::DataType::FLOAT32: {
            maybe_convert_tensor_dtype(torch::kFloat32);
            return create_concrete<float>(contiguous_tensor, tensor_spec);
        }
        case ttnn::DataType::BFLOAT16: {
            maybe_convert_tensor_dtype(at::kBFloat16);
            return create_concrete<bfloat16>(contiguous_tensor, tensor_spec);
        }        
        case ttnn::DataType::BFLOAT8_B:
        case ttnn::DataType::BFLOAT4_B: {
            maybe_convert_tensor_dtype(at::kFloat);
            return create_concrete<float>(contiguous_tensor, tensor_spec);
        }
        default: {
            TT_THROW("Unsupported DataType: {}", static_cast<int>(data_type));
        }
    }
}

torch::Tensor to_torch(const ttnn::Tensor& tensor, const bool padded_output) {
    auto logical_shape = tensor.logical_shape();
    auto data_type = tensor.dtype();
    auto torch_dtype = torch::kFloat;
    auto tensor_spec = tensor.tensor_spec();
    auto view = logical_shape.view();
    std::vector<int64_t> torch_shape(view.begin(), view.end());

    auto host_buffer = get_host_buffer_from_tensor(tensor, padded_output);

    void* data_ptr = nullptr;
    switch (data_type) {
        case ttnn::DataType::UINT8: {
            torch_dtype = torch::kUInt8;
            data_ptr = host_buffer.view_as<uint8_t>().data();
            break;
        }
        case ttnn::DataType::UINT16: {
            torch_dtype = torch::kUInt16;
            data_ptr = host_buffer.view_as<uint16_t>().data();
            break;
        }
        case ttnn::DataType::INT32: {
            torch_dtype = torch::kInt32;
            data_ptr = host_buffer.view_as<int32_t>().data();
            break;
        }
        case ttnn::DataType::UINT32: {
            torch_dtype = torch::kUInt32;
            data_ptr = host_buffer.view_as<uint32_t>().data();
            break;
        }
        case ttnn::DataType::BFLOAT8_B:
        case ttnn::DataType::BFLOAT4_B:
        case ttnn::DataType::FLOAT32: {
            torch_dtype = torch::kFloat;
            data_ptr = host_buffer.view_as<float>().data();
            break;
        }
        case ttnn::DataType::BFLOAT16: {
            torch_dtype = torch::kBFloat16;
            data_ptr = host_buffer.view_as<bfloat16>().data();
            break;
        }
        default: {
            TT_THROW("Unsupported DataType: {}", static_cast<int>(data_type));
            break;
        }
    }
    auto torch_tensor = torch::empty(torch_shape, torch_dtype);
    torch_tensor.copy_(torch::from_blob(data_ptr, torch_shape, at::TensorOptions().dtype(torch_dtype).requires_grad(false)));
    // If the tensor is padded, we need to reshape it to the padded shape
    if (padded_output) {
        auto shape = tensor.padded_shape();
        torch_shape = std::vector<int64_t>{shape.cbegin(), shape.cend()};
    }
    torch_tensor = torch_tensor.reshape(torch_shape);
    return torch_tensor.contiguous();
}


ttnn::Tensor apply_layernorm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias,
    float eps
) {
    // Apply layer normalization using ttnn operations
    return ttnn::layer_norm(input, eps, weight, bias);
}

/**
 * Load and preprocess image for DeiT model inference
 * Mimics the functionality of AutoImageProcessor from transformers
 * 
 * @param image_path Path to the input image file
 * @param device Target mesh device for tensor operations
 * @return Preprocessed image tensor [1, 3, 224, 224] ready for DeiT inference
 */
torch::Tensor load_and_preprocess_image(
    const std::string& image_path
) {
    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image from path: " + image_path);
    }
    
    // Convert BGR to RGB (OpenCV loads as BGR by default)
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    // Resize to 224x224 (DeiT input size)
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    // Create torch tensor from OpenCV Mat
    // Shape: [H, W, C] -> [C, H, W]
    torch::Tensor tensor = torch::from_blob(
        float_image.data, 
        {224, 224, 3}, 
        torch::kFloat32
    ).clone();
    
    // Permute dimensions from HWC to CHW
    tensor = tensor.permute({2, 0, 1});
    
    // Apply ImageNet normalization
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
    tensor = (tensor - mean) / std;
    
    // Add batch dimension: [C, H, W] -> [1, C, H, W]
    tensor = tensor.unsqueeze(0);
    
    // Return torch tensor directly
    return tensor;
}

/**
 * Compute Pearson Correlation Coefficient (PCC) between two tensors
 * @param tensor1 First tensor
 * @param tensor2 Second tensor
 * @return PCC value
 */
double compute_pcc(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
    auto flat1 = tensor1.flatten().to(torch::kFloat32);
    auto flat2 = tensor2.flatten().to(torch::kFloat32);
    
    auto mean1 = flat1.mean();
    auto mean2 = flat2.mean();
    
    auto centered1 = flat1 - mean1;
    auto centered2 = flat2 - mean2;
    
    auto numerator = (centered1 * centered2).sum();
    auto denominator = torch::sqrt((centered1 * centered1).sum() * (centered2 * centered2).sum());
    
    return numerator.item<double>() / denominator.item<double>();
}

} // namespace helper_funcs