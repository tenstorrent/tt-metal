// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "frobenius_normalize_device_operation.hpp"

#include "frobenius_normalize_program_factory.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

void FrobeniusNormalizeDeviceOperation::validate_on_program_cache_miss(
    const FrobeniusNormalizeAttributes& args, const FrobeniusNormalizeTensorArgs& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    check_device_tensor(input_tensor, "FrobeniusNormalize", "input");

    if (tensor_args.preallocated_output.has_value()) {
        const auto& output = tensor_args.preallocated_output.value();
        check_device_tensor(output, "FrobeniusNormalize", "preallocated_output");
        check_same_device(output, input_tensor, "FrobeniusNormalize", "preallocated_output", "input");
        TT_FATAL(output.logical_shape() == input_tensor.logical_shape(), "Preallocated output shape must match input.");
    }
}

FrobeniusNormalizeSpecReturn FrobeniusNormalizeDeviceOperation::compute_output_specs(
    const FrobeniusNormalizeAttributes& args, const FrobeniusNormalizeTensorArgs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return {tensor_args.preallocated_output->tensor_spec()};
    }

    return {tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()))};
}

FrobeniusNormalizeTensorReturn FrobeniusNormalizeDeviceOperation::create_output_tensors(
    const FrobeniusNormalizeAttributes& args, const FrobeniusNormalizeTensorArgs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return {tensor_args.preallocated_output.value()};
    }

    auto output_specs = compute_output_specs(args, tensor_args);
    return {ttnn::create_device_tensor(output_specs[0], tensor_args.input.device())};
}

ttsl::hash::hash_t FrobeniusNormalizeDeviceOperation::compute_program_hash(
    const FrobeniusNormalizeAttributes& args, const FrobeniusNormalizeTensorArgs& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<FrobeniusNormalizeDeviceOperation>(
        input_tensor.dtype(), input_tensor.logical_shape());
}

}  // namespace ttml::metal::ops::frobenius_normalize::device

namespace ttnn::prim {

ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeTensorReturn ttml_frobenius_normalize(
    const ttnn::Tensor& input_tensor, float epsilon, const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeDeviceOperation;

    namespace fn_device = ttml::metal::ops::frobenius_normalize::device;
    auto operation_attributes = fn_device::FrobeniusNormalizeAttributes{.epsilon = epsilon};
    auto tensor_args = fn_device::FrobeniusNormalizeTensorArgs{
        .input = input_tensor,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
