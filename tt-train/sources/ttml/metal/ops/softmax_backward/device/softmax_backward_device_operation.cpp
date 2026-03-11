// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::softmax_backward::device {

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t&) {
    TT_THROW("softmax_backward device operation has been removed");
}

SoftmaxBackwardDeviceOperation::spec_return_value_t SoftmaxBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.softmax_output.tensor_spec();
}

SoftmaxBackwardDeviceOperation::tensor_return_value_t SoftmaxBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs({}, tensor_args), tensor_args.softmax_output.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SoftmaxBackwardDeviceOperation::tensor_return_value_t>
SoftmaxBackwardDeviceOperation::create_op_performance_model(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    return {};
}

}  // namespace ttml::metal::ops::softmax_backward::device

namespace ttnn::prim {

ttnn::Tensor ttml_softmax_backward(
    const ttnn::Tensor&, const ttnn::Tensor&, uint32_t, const std::optional<tt::tt_metal::CoreRangeSet>&) {
    throw std::runtime_error("softmax_backward device operation has been removed");
}

}  // namespace ttnn::prim
