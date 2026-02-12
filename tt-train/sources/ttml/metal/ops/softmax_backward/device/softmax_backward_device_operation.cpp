// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_device_operation.hpp"

#include "tt_stl/assert.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttml::metal::ops::softmax_backward::device {

SoftmaxBackwardDeviceOperation::program_factory_t SoftmaxBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return SoftmaxBackwardFactory{};
}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;
    TT_FATAL(
        softmax_output.logical_shape() == upstream_grad.logical_shape(),
        "Softmax output and upstream gradient tensors must have the same shape");
    TT_FATAL(softmax_output.dtype() == DataType::BFLOAT16, "Softmax backward only supports BFLOAT16");
    TT_FATAL(
        upstream_grad.dtype() == softmax_output.dtype(),
        "Softmax output and upstream gradient must have the same dtype");
    TT_FATAL(softmax_output.layout() == Layout::TILE, "Softmax backward requires TILE layout");
    TT_FATAL(upstream_grad.layout() == Layout::TILE, "Softmax backward requires TILE layout");
    const auto rank = softmax_output.logical_shape().rank();
    TT_FATAL(
        attributes.dim == rank - 1,
        "Currently only supporting softmax_backward on last dimension (got dim={}, rank={})",
        attributes.dim,
        rank);
}

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;
    const ttnn::Tensor& upstream_grad = tensor_args.upstream_grad;
    TT_FATAL(
        softmax_output.logical_shape() == upstream_grad.logical_shape(),
        "Softmax output and upstream gradient tensors must have the same shape");
}

SoftmaxBackwardDeviceOperation::spec_return_value_t SoftmaxBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.softmax_output;
    return {
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), input_tensor.memory_config())};
}

SoftmaxBackwardDeviceOperation::tensor_return_value_t SoftmaxBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.softmax_output.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SoftmaxBackwardDeviceOperation::tensor_return_value_t>
SoftmaxBackwardDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;
    const auto& output_tensor = tensor_return_value;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(softmax_output, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {softmax_output, upstream_grad}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttml::metal::ops::softmax_backward::device

namespace ttnn::prim {

ttnn::Tensor ttml_softmax_backward(
    const ttnn::Tensor& softmax_output, const ttnn::Tensor& upstream_grad, uint32_t dim) {
    using OperationType = ttml::metal::ops::softmax_backward::device::SoftmaxBackwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{dim};
    auto tensor_args = OperationType::tensor_args_t{softmax_output, upstream_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
