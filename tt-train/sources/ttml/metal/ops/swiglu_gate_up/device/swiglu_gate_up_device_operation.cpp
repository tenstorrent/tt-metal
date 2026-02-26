// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_gate_up_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_gate_up::device {

void SwiGLUGateUpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SwiGLUGateUp operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to SwiGLUGateUp need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SwiGLUGateUp operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SwiGLUGateUp operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SwiGLUGateUp operation requires Interleaved memory layout. {} memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w3 = tensor_args.w3;

    check_tensor(input_tensor, "Input");
    check_tensor(w1, "W1");
    check_tensor(w3, "W3");

    const auto& input_shape = input_tensor.logical_shape();
    const auto& w1_shape = w1.logical_shape();
    const auto& w3_shape = w3.logical_shape();

    TT_FATAL(input_shape.rank() == 4U, "Input tensor must be 4D");

    uint32_t embed_dim = input_shape[-1];
    uint32_t hidden_dim = w1_shape[-1];

    TT_FATAL(
        w1_shape[-2] == embed_dim, "W1 shape mismatch: W1[-2]={} must equal input[-1]={}.", w1_shape[-2], embed_dim);
    TT_FATAL(
        w3_shape[-2] == embed_dim && w3_shape[-1] == hidden_dim,
        "W3 shape mismatch: W3={} must match W1={}. Both should be [embed, hidden].",
        w3_shape,
        w1_shape);

    TT_FATAL(embed_dim % tt::constants::TILE_WIDTH == 0, "Embed dimension must be multiple of TILE_WIDTH");
    TT_FATAL(hidden_dim % tt::constants::TILE_WIDTH == 0, "Hidden dimension must be multiple of TILE_WIDTH");
}

spec_return_value_t SwiGLUGateUpDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_shape = tensor_args.input.logical_shape();
    const auto& w1_shape = tensor_args.w1.logical_shape();

    // Output M has shape [B, 1, S, hidden_dim]
    ttnn::Shape output_shape({input_shape[0], input_shape[1], input_shape[2], w1_shape[3]});

    spec_return_value_t output_specs;
    output_specs.reserve(1U);
    output_specs.emplace_back(
        output_shape,
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));

    return output_specs;
}

tensor_return_value_t SwiGLUGateUpDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_specs[0], tensor_args.input.device());
}

ttsl::hash::hash_t SwiGLUGateUpDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w3 = tensor_args.w3;
    return tt::tt_metal::operation::hash_operation<SwiGLUGateUpDeviceOperation>(
        args, input.dtype(), input.logical_shape(), w1.dtype(), w1.logical_shape(), w3.dtype(), w3.logical_shape());
}

}  // namespace ttml::metal::ops::swiglu_gate_up::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_gate_up::device::SwiGLUGateUpDeviceOperation::tensor_return_value_t ttml_swiglu_gate_up(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w3) {
    using OperationType = ttml::metal::ops::swiglu_gate_up::device::SwiGLUGateUpDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .w1 = w1, .w3 = w3};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
