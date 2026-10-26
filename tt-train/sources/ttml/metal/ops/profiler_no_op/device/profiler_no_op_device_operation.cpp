// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "profiler_no_op_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "profiler_no_op_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::profiler_no_op::device {

void ProfilerNoopOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_device_tensor(input_tensor, "ProfilerNoop", "Input", {.layout = tt::tt_metal::Layout::ROW_MAJOR});
    if (preallocated_output_tensor.has_value()) {
        check_device_tensor(preallocated_output_tensor.value(), "ProfilerNoop", "Preallocated Output");
    }
}

ProfilerNoopOperation::spec_return_value_t ProfilerNoopOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    auto input_logical_shape = tensor_args.input.logical_shape();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape(input_logical_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
}

ProfilerNoopOperation::tensor_return_value_t ProfilerNoopOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensor = tensor_args.preallocated_output.value();
    } else {
        output_tensor = ttnn::create_device_tensor(output_specs, tensor_args.input.device());
    }

    return output_tensor;
}

ttsl::hash::hash_t ProfilerNoopOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    return tt::tt_metal::operation::hash_operation<ProfilerNoopOperation>(
        args, input_tensor.dtype(), input_logical_shape);
}

}  // namespace ttml::metal::ops::profiler_no_op::device

namespace ttnn::prim {

ttml::metal::ops::profiler_no_op::device::ProfilerNoopOperation::tensor_return_value_t ttml_profiler_no_op(
    const ttnn::Tensor& input_tensor,
    const std::string& identifier,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::profiler_no_op::device::ProfilerNoopOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.identifier = identifier};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
