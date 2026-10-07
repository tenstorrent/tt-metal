// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_bw_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

void GatedRmsNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    (void)validate_and_get_geometry(
        "gated_rmsnorm_bw", tensor_args.input, tensor_args.gate, tensor_args.gamma, tensor_args.dL_dout);
}

bw::spec_return_value_t GatedRmsNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto spec = tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.input.memory_config()));
    bw::spec_return_value_t specs{spec, spec, std::nullopt};
    if (args.compute_dgamma) {
        specs[2] = spec;
    }
    return specs;
}

bw::tensor_return_value_t GatedRmsNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    bw::tensor_return_value_t outputs(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].has_value()) {
            outputs[i] = ttnn::create_device_tensor(*specs[i], tensor_args.input.device());
        }
    }
    return outputs;
}

ttsl::hash::hash_t GatedRmsNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<GatedRmsNormBackwardDeviceOperation>(
        args.epsilon, args.compute_dgamma, tensor_args.input.logical_shape(), tensor_args.gamma.logical_shape());
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device

namespace ttnn::prim {

ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormBackwardDeviceOperation::tensor_return_value_t
ttml_gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    float epsilon,
    bool compute_dgamma) {
    using OperationType = ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormBackwardDeviceOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{.epsilon = epsilon, .compute_dgamma = compute_dgamma};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .gate = gate, .gamma = gamma, .dL_dout = dL_dout};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
