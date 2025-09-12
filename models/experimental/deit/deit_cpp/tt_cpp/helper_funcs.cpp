#include "helper_funcs.h"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <torch/torch.h>
#include <vector>
#include <stdexcept>

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
    if (bias.has_value() && bias->get_layout() != ttnn::TILE_LAYOUT) {
        bias = ttnn::to_layout(bias.value(), ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (ttnn::MeshDevice*)nullptr);
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
        TT_THROW("from_torch Unsurport type : {}", c10::toString(torch_dtype));
    }

    if (data_type == ttnn::DataType::BFLOAT8_B || data_type == ttnn::DataType::BFLOAT4_B) {
        throw std::runtime_error("from_torch: bfloat8_b/bfloat4_b unsurport!");
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

} // namespace helper_funcs