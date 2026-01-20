// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "helper_funcs.h"
#include <stdexcept>
#include <memory>
#include <iostream>
#include <algorithm>
#include <fmt/format.h>
#include <stdlib.h>
#include "ttnn_mobilenetv2.h"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/distributed/api.hpp"

template <typename T>
ttnn::Tensor create_concrete(torch::Tensor& contiguous_tensor, tt::tt_metal::TensorSpec& spec) {
    return ttnn::Tensor::template from_span<T>(
        tt::stl::template Span<const T>(reinterpret_cast<T*>(contiguous_tensor.data_ptr()), contiguous_tensor.numel()),
        spec);
}

ttnn::Tensor from_torch(
    const at::Tensor& tensor, std::optional<ttnn::DataType> dtype, std::optional<ttnn::Layout> layout) {
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

    auto tensor_spec = tt::tt_metal::TensorSpec(
        logical_shape,
        tt::tt_metal::TensorLayout(
            data_type, tt::tt_metal::PageConfig(layout.value_or(ttnn::Layout::ROW_MAJOR)), ttnn::MemoryConfig{}));
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
                tt::stl::Span<const std::uint32_t> uint32_data =
                    tt::tt_metal::host_buffer::get_as<std::uint32_t>(buffer);
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

torch::Tensor to_torch(const ttnn::Tensor& tensor, const bool padded_output) {
    const auto& logical_shape = tensor.logical_shape();
    auto data_type = tensor.dtype();
    auto torch_dtype = torch::kFloat;
    // auto tensor_spec = tensor.tensor_spec();
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
    torch_tensor.copy_(
        torch::from_blob(data_ptr, torch_shape, at::TensorOptions().dtype(torch_dtype).requires_grad(false)));
    // If the tensor is padded, we need to reshape it to the padded shape
    if (padded_output) {
        const auto& shape = tensor.padded_shape();
        torch_shape = std::vector<int64_t>{shape.cbegin(), shape.cend()};
    }
    torch_tensor = torch_tensor.reshape(torch_shape);
    return torch_tensor.contiguous();
}

// Helper function to create input tensors
std::tuple<at::Tensor, ttnn::Tensor> create_mobilenetv2_input_tensors(
    int batch, int input_channels, int input_height, int input_width) {
    torch::Tensor torch_input_tensor = torch::randn({batch, input_channels, input_height, input_width});

    torch::Tensor permuted_tensor = torch_input_tensor.permute({0, 2, 3, 1});
    torch::Tensor reshaped_tensor = permuted_tensor.reshape(
        {1, 1, permuted_tensor.size(0) * permuted_tensor.size(1) * permuted_tensor.size(2), permuted_tensor.size(3)});

    ttnn::Tensor ttnn_input_tensor = from_torch(reshaped_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    return std::make_tuple(torch_input_tensor, ttnn_input_tensor);
}

// Helper function to fold BatchNorm2d into Conv2d
std::pair<ttnn::Tensor, ttnn::Tensor> fold_batch_norm2d_into_conv2d(
    const torch::jit::Module& conv, const torch::jit::Module& bn) {
    float eps = 1e-5;  // Default epsilon value for numerical stability
    auto weight = conv.attr("weight").toTensor().detach();
    auto running_mean = bn.attr("running_mean").toTensor().detach();
    auto running_var = bn.attr("running_var").toTensor().detach();
    auto scale = bn.attr("weight").toTensor().detach();
    auto shift = bn.attr("bias").toTensor().detach();

    weight = weight * (scale / torch::sqrt(running_var + eps)).view({-1, 1, 1, 1});
    auto bias = shift - running_mean * (scale / torch::sqrt(running_var + eps));
    bias = bias.view({1, 1, 1, -1});

    ttnn::Tensor weight_ttnn = from_torch(weight, ttnn::DataType::FLOAT32);
    ttnn::Tensor bias_ttnn = from_torch(bias, ttnn::DataType::FLOAT32);

    return std::make_pair(weight_ttnn, bias_ttnn);
}

ttnn::Tensor preprocess_linear_weight(
    const at::Tensor& weight,
    std::optional<ttnn::DataType> dtype = std::nullopt,
    ttnn::Layout layout = ttnn::Layout::TILE) {
    return from_torch(weight.transpose(0, 1).contiguous().detach(), dtype, layout);
}

ttnn::Tensor preprocess_linear_bias(
    const at::Tensor& bias,
    std::optional<ttnn::DataType> dtype = std::nullopt,
    ttnn::Layout layout = ttnn::Layout::TILE) {
    return from_torch(bias.reshape({1, -1}).detach(), dtype, layout);
}

#define MODULE_NAME(m) (m).type()->str()

// Function to create MobileNetV2 model parameters
std::unordered_map<std::string, ttnn::Tensor> create_mobilenetv2_model_parameters(
    const torch::jit::Module& model, const std::shared_ptr<ttnn::MeshDevice>& device) {
    std::unordered_map<std::string, ttnn::Tensor> model_parameters;
    int conv_bn_counter = 0;
    int counter = 0;

    for (const auto& child : model.children()) {
        if (MODULE_NAME(child).find("Sequential") != std::string::npos) {
            for (const auto& seq_child : child.children()) {
                if (MODULE_NAME(seq_child).find("Conv2dNormActivation") != std::string::npos) {
                    auto conv_itor = seq_child.children().begin();
                    auto conv = *conv_itor;
                    ++conv_itor;
                    auto bn = *conv_itor;
                    auto [weight_ttnn, bias_ttnn] = fold_batch_norm2d_into_conv2d(conv, bn);
                    model_parameters[fmt::format("fused_conv_{}_weight", conv_bn_counter)] = weight_ttnn;
                    model_parameters[fmt::format("fused_conv_{}_bias", conv_bn_counter)] = bias_ttnn;
                    conv_bn_counter++;
                } else if (MODULE_NAME(seq_child).find("InvertedResidual") != std::string::npos) {
                    auto invert_seq = *(seq_child.children().begin());
                    assert(MODULE_NAME(invert_seq).find("Sequential") != std::string::npos);
                    auto invert_module_list = invert_seq.children();
                    auto invert_seq_itor = invert_module_list.begin();
                    do {
                        if (MODULE_NAME(*invert_seq_itor).find("Conv2dNormActivation") != std::string::npos) {
                            auto conv_itor = (*invert_seq_itor).children().begin();
                            auto conv = *conv_itor;
                            ++conv_itor;
                            auto bn = *conv_itor;
                            auto [weight_ttnn, bias_ttnn] = fold_batch_norm2d_into_conv2d(conv, bn);
                            model_parameters[fmt::format("fused_conv_{}_weight", conv_bn_counter)] = weight_ttnn;
                            model_parameters[fmt::format("fused_conv_{}_bias", conv_bn_counter)] = bias_ttnn;
                            conv_bn_counter++;
                        } else if (MODULE_NAME(*invert_seq_itor).find("Conv2d") != std::string::npos) {
                            auto conv2d = *invert_seq_itor;
                            ++invert_seq_itor;
                            if (invert_seq_itor != invert_module_list.end()) {
                                auto bn = *invert_seq_itor;
                                if (MODULE_NAME(bn).find("BatchNorm2d") != std::string::npos) {
                                    auto [weight_ttnn, bias_ttnn] = fold_batch_norm2d_into_conv2d(conv2d, bn);
                                    model_parameters[fmt::format("conv_{}_weight", counter)] = weight_ttnn;
                                    model_parameters[fmt::format("conv_{}_bias", counter)] = bias_ttnn;
                                    ++counter;
                                }
                            }
                        } else {
                            TT_THROW("Unsupported InvertedResidual Children Module {}", MODULE_NAME(*invert_seq_itor));
                        }
                        ++invert_seq_itor;
                    } while (invert_seq_itor != invert_module_list.end());
                } else if (MODULE_NAME(seq_child).find("Linear") != std::string::npos) {
                    auto weight =
                        preprocess_linear_weight(seq_child.attr("weight").toTensor().detach(), ttnn::DataType::FLOAT32);
                    auto bias =
                        preprocess_linear_bias(seq_child.attr("bias").toTensor().detach(), ttnn::DataType::FLOAT32);
                    model_parameters["classifier_1_weight"] = weight.to_device(device.get());
                    model_parameters["classifier_1_bias"] = bias.to_device(device.get());
                } else {
                    TT_THROW("Unsupported Module {}", MODULE_NAME(seq_child));
                }
            }
        }
    }
    return model_parameters;
}

// Helper function to get buffer addresses
uint32_t get_ttbuffer_address(const ttnn::Tensor& tensor) {
    return std::visit(
        tt::stl::overloaded{
            [](const tt::tt_metal::DeviceStorage& s) -> uint32_t {
                if (s.mesh_buffer) {
                    return s.mesh_buffer->address();
                } else {
                    TT_THROW("Tensor is not allocated.");
                }
            },
            [](auto&&) -> uint32_t {
                TT_THROW("HostStorage doesn't support buffer_address method");
                return 0;
            }},
        tensor.storage());
}

std::tuple<bool, double> comp_pcc(
    const torch::Tensor& golden, const torch::Tensor& calculated, const double pcc_threshold) {
    torch::Tensor mut_calculated(calculated);
    if (golden.dtype() != mut_calculated.dtype()) {
        mut_calculated = mut_calculated.to(golden.dtype());
    }

    if (torch::all(golden.isnan()).item<bool>() && torch::all(mut_calculated.isnan()).item<bool>()) {
        return std::make_tuple(true, 1.0);
    }

    if (torch::all(golden.isnan()).item<bool>() || torch::all(mut_calculated.isnan()).item<bool>()) {
        return std::make_tuple(false, 0.0);
    }

    if (!torch::any(golden.to(torch::kBool)).equal(torch::any(mut_calculated.to(torch::kBool)))) {
        return std::make_tuple(false, 0.0);
    }

    auto mut_golden = golden.masked_fill(
        torch::logical_or(golden.isnan(), torch::logical_or(golden.isneginf(), golden.isinf())), 0.0);
    mut_calculated = mut_calculated.masked_fill(
        torch::logical_or(mut_calculated.isnan(), torch::logical_or(mut_calculated.isneginf(), mut_calculated.isinf())),
        0.0);

    if (torch::equal(mut_golden, mut_calculated)) {
        return std::make_tuple(true, 1.0);
    }
    auto pcc = torch::stack({mut_golden.squeeze().flatten(), mut_calculated.squeeze().flatten()})
                   .corrcoef()
                   .min()
                   .item<double>();
    return std::make_tuple(pcc >= pcc_threshold, pcc);
}

std::string assert_with_pcc(const at::Tensor& golden, const at::Tensor& calculated, const double pcc_threshold) {
    auto [is_passed, pcc] = comp_pcc(golden, calculated, pcc_threshold);
    if (is_passed) {
        return fmt::format("PCC= {:.4f} - Passed", pcc);
    } else {
        return fmt::format("PCC= {:.4f} - Failed", pcc);
    }
}

// Utility functions
uint32_t divup(uint32_t x, uint32_t y) { return static_cast<uint32_t>((x + y - 1) / y); }

bool isWormholeB0() {
    // Hypothetical function to check if the device is Wormhole B0
    return true;
}

// Load Torch model
torch::jit::Module loadTorchModel() {
    std::string model_file;
    if (auto path = getenv("MOBILENET_FILE_PATH"); path != nullptr) {
        model_file.assign(path);
    } else {
        const char* metal_path = getenv("TT_METAL_HOME");
        TT_FATAL(metal_path != nullptr, "TT_METAL_HOME must be set!!!");
        model_file = fmt::format("{}/models/mobilenetv2_cpp/mobilenet_v2-b0353104-script.pt", metal_path);
    }
    auto torch_model = torch::jit::load(model_file);
    torch_model.eval();
    return torch_model;
}

// Load TTNN model
std::shared_ptr<TtMobileNetV2> loadTtnnModel(
    const std::shared_ptr<ttnn::MeshDevice>& device, const torch::jit::Module& torch_model, int batch_size) {
    auto model_parameters = create_mobilenetv2_model_parameters(torch_model, device);
    auto ttnn_model = std::make_shared<TtMobileNetV2>(model_parameters, device, batch_size);
    return ttnn_model;
}
