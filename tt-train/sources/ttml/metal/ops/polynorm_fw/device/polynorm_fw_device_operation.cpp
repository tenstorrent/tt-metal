// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

void PolyNorm3ForwardDeviceOperation::validate_on_program_cache_miss(
    const PolyNorm3FWAttributes& args, const PolyNorm3FWTensorArgs& tensor_args) {
    check_device_tensor(tensor_args.input, "PolyNorm3Forward", "Input");
    check_device_tensor(tensor_args.weight, "PolyNorm3Forward", "weight");
    check_device_tensor(tensor_args.bias, "PolyNorm3Forward", "bias");
    if (tensor_args.preallocated_output.has_value()) {
        check_device_tensor(tensor_args.preallocated_output.value(), "PolyNorm3Forward", "Preallocated output");
    }
}

PolyNorm3FWSpecReturn PolyNorm3ForwardDeviceOperation::compute_output_specs(
    const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return {tensor_args.preallocated_output->tensor_spec()};
    }
    return {tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()))};
}

PolyNorm3FWTensorReturn PolyNorm3ForwardDeviceOperation::create_output_tensors(
    const PolyNorm3FWAttributes& op_attrs, const PolyNorm3FWTensorArgs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    auto specs = compute_output_specs(op_attrs, tensor_args);
    return ttnn::create_device_tensor(specs[0], tensor_args.input.device());
}

}  // namespace ttml::metal::ops::polynorm3_fw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm3_fw::device::PolyNorm3ForwardDeviceOperation::tensor_return_value_t ttml_polynorm3_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::polynorm3_fw::device::PolyNorm3ForwardDeviceOperation;

    const auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
    };
    const auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .weight = weight,
        .bias = bias,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
