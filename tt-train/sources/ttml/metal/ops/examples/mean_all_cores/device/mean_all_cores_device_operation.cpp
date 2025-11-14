// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mean_all_cores_device_operation.hpp"

#include "mean_all_cores_program_factory.hpp"

#include <enchantum/enchantum.hpp>

namespace ttml::metal::ops::examples::mean_all_cores::device {

MeanAllCoresDeviceOperation::program_factory_t MeanAllCoresDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return MeanAllCoresProgramFactory{};
}

void MeanAllCoresDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MeanAllCoresDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    TT_FATAL(
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "MeanAllCores operation requires 'input' to be on DEVICE. Got storage type: '{}'",
        enchantum::to_string(input_tensor.storage_type()));

    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated on device (buffer is null).");

    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor must have TILE layout, but got '{}'",
        enchantum::to_string(input_tensor.layout()));

    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Input tensor must have BFLOAT16 data type, but got '{}'",
        enchantum::to_string(input_tensor.dtype()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
        "Input tensor must use INTERLEAVED memory layout, but got '{}'",
        enchantum::to_string(input_tensor.memory_config().memory_layout()));

    if (preallocated_output_tensor.has_value()) {
        TT_FATAL(
            preallocated_output_tensor->storage_type() == tt::tt_metal::StorageType::DEVICE,
            "Preallocated output tensor must be on DEVICE");
        TT_FATAL(
            preallocated_output_tensor->layout() == tt::tt_metal::Layout::TILE,
            "Preallocated output tensor must have TILE layout");
        TT_FATAL(
            preallocated_output_tensor->dtype() == tt::tt_metal::DataType::BFLOAT16,
            "Preallocated output tensor must have BFLOAT16 data type");
    }
}

MeanAllCoresDeviceOperation::spec_return_value_t MeanAllCoresDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    // Output shape is (1, 1, 1, 1) for mean result
    ttnn::Shape output_shape({1, 1, 1, 1});
    return ttnn::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
}

MeanAllCoresDeviceOperation::tensor_return_value_t MeanAllCoresDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensor = tensor_args.preallocated_output.value();
    } else {
        output_tensor = create_device_tensor(output_specs, tensor_args.input.device());
    }

    return output_tensor;
}

ttsl::hash::hash_t MeanAllCoresDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    auto hash = tt::tt_metal::operation::hash_operation<MeanAllCoresDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

std::tuple<operation_attributes_t, tensor_args_t> MeanAllCoresDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    return {
        operation_attributes_t{},
        tensor_args_t{
            .input = input_tensor,
            .preallocated_output = preallocated_output,
        }};
}

}  // namespace ttml::metal::ops::examples::mean_all_cores::device

