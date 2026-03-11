// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::softmax::device {

void SoftmaxDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {
    TT_THROW("softmax device operation has been removed");
}

SoftmaxDeviceOperation::spec_return_value_t SoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.input.tensor_spec();
}

SoftmaxDeviceOperation::tensor_return_value_t SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs({}, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t SoftmaxDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SoftmaxDeviceOperation>(
        args, tensor_args.input.dtype(), tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::softmax::device

namespace ttnn::prim {

ttml::metal::ops::softmax::device::SoftmaxDeviceOperation::tensor_return_value_t ttml_softmax(
    const ttnn::Tensor& input_tensor, int32_t dim, const std::optional<ttnn::Tensor>& preallocated_output) {
    throw std::runtime_error("softmax device operation has been removed");
}

}  // namespace ttnn::prim
