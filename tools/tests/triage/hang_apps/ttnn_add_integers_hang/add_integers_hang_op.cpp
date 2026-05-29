// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_integers_hang_op.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace triage_hang_apps {

AddIntegersHangOperation::program_factory_t AddIntegersHangOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void AddIntegersHangOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    TT_FATAL(
        a.logical_shape() == b.logical_shape(),
        "add_integers_hang: input tensors must have the same shape (got {} and {})",
        a.logical_shape(),
        b.logical_shape());
    TT_FATAL(
        a.dtype() == b.dtype(),
        "add_integers_hang: input tensors must have the same dtype (got {} and {})",
        a.dtype(),
        b.dtype());
}

AddIntegersHangOperation::spec_return_value_t AddIntegersHangOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    return ttnn::TensorSpec(
        a.logical_shape(),
        tt::tt_metal::TensorLayout(a.dtype(), tt::tt_metal::PageConfig(a.layout()), ttnn::MemoryConfig{}));
}

AddIntegersHangOperation::tensor_return_value_t AddIntegersHangOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

ttnn::Tensor add_integers_hang(const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b) {
    return ttnn::device_operation::launch<AddIntegersHangOperation>(
        AddIntegersHangOperation::operation_attributes_t{},
        AddIntegersHangOperation::tensor_args_t{input_tensor_a, input_tensor_b});
}

}  // namespace triage_hang_apps
