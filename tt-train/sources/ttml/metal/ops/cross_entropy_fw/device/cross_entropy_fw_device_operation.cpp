// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw_device_operation.hpp"

#include "cross_entropy_fw_program_factory.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

CrossEntropyForwardDeviceOperation::program_factory_t CrossEntropyForwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return CrossEntropyForwardProgramFactory{};
}

void CrossEntropyForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void CrossEntropyForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
            "CrossEntropyForward operation is only supported on Wormhole. Device arch: {}. Tensor name: {}",
            magic_enum::enum_name(tensor.device()->arch()),
            name);

        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "CrossEntropyForward operation requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            magic_enum::enum_name(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.get_layout() == required_layout,
            "Tensor '{}' must have layout '{}', but got '{}'",
            name,
            magic_enum::enum_name(required_layout),
            magic_enum::enum_name(tensor.get_layout()));

        TT_FATAL(
            tensor.get_dtype() == required_dtype,
            "Tensor '{}' must have data type '{}', but got '{}'",
            name,
            magic_enum::enum_name(required_dtype),
            magic_enum::enum_name(tensor.get_dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use INTERLEAVED memory layout, but got '{}'",
            name,
            magic_enum::enum_name(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& target_tensor = tensor_args.target;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_tensor(input_tensor, "Input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(target_tensor, "Target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);
    if (preallocated_output_tensor.has_value()) {
        check_tensor(
            preallocated_output_tensor.value(),
            "Preallocated Output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);
    }
}

CrossEntropyForwardDeviceOperation::spec_return_value_t CrossEntropyForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->get_tensor_spec();
    }
    auto input_logical_shape = tensor_args.input.get_logical_shape();
    input_logical_shape[-1] = 1U;
    return ttnn::TensorSpec(
        ttnn::Shape(input_logical_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.input.get_dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
}

CrossEntropyForwardDeviceOperation::tensor_return_value_t CrossEntropyForwardDeviceOperation::create_output_tensors(
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

tt::stl::hash::hash_t CrossEntropyForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.get_logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    auto hash = tt::tt_metal::operation::hash_operation<CrossEntropyForwardDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

std::tuple<operation_attributes_t, tensor_args_t> CrossEntropyForwardDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& target_tensor,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    return {
        operation_attributes_t{},
        tensor_args_t{
            .input = input_tensor,
            .target = target_tensor,
            .preallocated_output = preallocated_output,
        }};
}

}  // namespace ttml::metal::ops::cross_entropy_fw::device
