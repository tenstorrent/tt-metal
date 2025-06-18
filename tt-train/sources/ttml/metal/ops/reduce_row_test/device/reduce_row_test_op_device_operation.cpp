// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_row_test_op_device_operation.hpp"

#include "reduce_row_test_op_program_factory.hpp"

namespace ttml::metal::ops::reduce_row_test_op::device {

ReduceRowTestDeviceOperation::program_factory_t ReduceRowTestDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ReduceRowTestProgramFactory{};
}

void ReduceRowTestDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ReduceRowTestDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
            "ReduceRowTest operation is only supported on Wormhole. Device arch: {}. Tensor name: {}",
            magic_enum::enum_name(tensor.device()->arch()),
            name);

        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "ReduceRowTest operation requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            magic_enum::enum_name(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.layout() == required_layout,
            "Tensor '{}' must have layout '{}', but got '{}'",
            name,
            magic_enum::enum_name(required_layout),
            magic_enum::enum_name(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == required_dtype,
            "Tensor '{}' must have data type '{}', but got '{}'",
            name,
            magic_enum::enum_name(required_dtype),
            magic_enum::enum_name(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use INTERLEAVED memory layout, but got '{}'",
            name,
            magic_enum::enum_name(tensor.memory_config().memory_layout()));
    };

    const auto& input = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_tensor(input, "Input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);

    if (preallocated_output_tensor.has_value()) {
        check_tensor(
            preallocated_output_tensor.value(),
            "Preallocated Output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);
    }
}

ReduceRowTestDeviceOperation::spec_return_value_t ReduceRowTestDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    auto input_logical_shape = tensor_args.input.logical_shape();
    input_logical_shape[-1] = 1;
    return ttnn::TensorSpec(
        ttnn::Shape(input_logical_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
}

ReduceRowTestDeviceOperation::tensor_return_value_t ReduceRowTestDeviceOperation::create_output_tensors(
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

tt::stl::hash::hash_t ReduceRowTestDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_logical_shape = input.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    return tt::tt_metal::operation::hash_operation<ReduceRowTestDeviceOperation>(
        args, program_factory.index(), input.dtype(), input_logical_shape);
}

std::tuple<operation_attributes_t, tensor_args_t> ReduceRowTestDeviceOperation::invoke(
    const ttnn::Tensor& input, const bool use_matmul, const std::optional<ttnn::Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            .use_matmul = use_matmul,
        },
        tensor_args_t{
            .input = input,
            .preallocated_output = preallocated_output,
        }};
}

}  // namespace ttml::metal::ops::reduce_row_test_op::device
