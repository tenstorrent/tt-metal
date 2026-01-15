// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_fw::device {

SwiGLUForwardDeviceOperation::program_factory_t SwiGLUForwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return SwiGLUForwardProgramFactory{};
}

void SwiGLUForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SwiGLUForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SwiGLUForward operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to SwiGLUForward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SwiGLUForward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SwiGLUForward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SwiGLUForward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w2 = tensor_args.w2;
    const auto& w3 = tensor_args.w3;
    const auto& preallocated_swiglu_tensor = tensor_args.preallocated_swiglu;

    check_tensor(input_tensor, "Input");
    check_tensor(w1, "W1");
    check_tensor(w2, "W2");
    check_tensor(w3, "W3");
    if (preallocated_swiglu_tensor.has_value()) {
        check_tensor(preallocated_swiglu_tensor.value(), "Preallocated SwiGLU");
    }
}

spec_return_value_t SwiGLUForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U);

    if (tensor_args.preallocated_swiglu.has_value()) {
        output_specs.push_back(tensor_args.preallocated_swiglu->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    return output_specs;
}

tensor_return_value_t SwiGLUForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_swiglu.has_value()) {
        return tensor_args.preallocated_swiglu.value();
    } else {
        return create_device_tensor(output_specs[0], tensor_args.input.device());
    }
}

ttsl::hash::hash_t SwiGLUForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_logical_shape = input.logical_shape();
    const auto& w1 = tensor_args.w1;
    const auto& w1_logical_shape = w1.logical_shape();
    const auto& w2 = tensor_args.w2;
    const auto& w2_logical_shape = w2.logical_shape();
    const auto& w3 = tensor_args.w3;
    const auto& w3_logical_shape = w3.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<SwiGLUForwardDeviceOperation>(
        args,
        program_factory.index(),
        input.dtype(),
        input_logical_shape,
        w1.dtype(),
        w1_logical_shape,
        w2.dtype(),
        w2_logical_shape,
        w3.dtype(),
        w3_logical_shape);

    return hash;
}

}  // namespace ttml::metal::ops::swiglu_fw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_fw::device::SwiGLUForwardDeviceOperation::tensor_return_value_t ttml_swiglu_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& m1,
    const ttnn::Tensor& m2,
    const ttnn::Tensor& m3,
    const std::optional<ttnn::Tensor>& preallocated_swiglu) {
    using OperationType = ttml::metal::ops::swiglu_fw::device::SwiGLUForwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor, .w1 = m1, .w2 = m2, .w3 = m3, .preallocated_swiglu = preallocated_swiglu};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
