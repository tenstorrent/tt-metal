// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_bw_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

void GatedRmsNormBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    validate_and_get_geometry(
        tensor_args.input, tensor_args.gate, tensor_args.gamma, tensor_args.dL_dout, "GatedRmsNormBw");
}

GatedRmsNormBwDeviceOperation::spec_return_value_t GatedRmsNormBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const tt::tt_metal::TensorSpec spec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, input.memory_config()));
    spec_return_value_t specs{spec, spec};
    if (args.compute_dgamma) {
        specs.push_back(spec);
    }
    return specs;
}

GatedRmsNormBwDeviceOperation::tensor_return_value_t GatedRmsNormBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* const device = tensor_args.input.device();
    tensor_return_value_t outputs;
    outputs.reserve(3U);
    for (const auto& spec : specs) {
        outputs.emplace_back(ttnn::create_device_tensor(spec, device));
    }
    if (!args.compute_dgamma) {
        outputs.emplace_back(std::nullopt);
    }
    return outputs;
}

ttsl::hash::hash_t GatedRmsNormBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<GatedRmsNormBwDeviceOperation>(
        args, tensor_args.input.logical_shape(), tensor_args.gamma.logical_shape());
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device

namespace ttnn::prim {

ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormBwDeviceOperation::tensor_return_value_t ttml_gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    const float epsilon,
    const bool compute_dgamma) {
    using Op = ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormBwDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{.epsilon = epsilon, .compute_dgamma = compute_dgamma},
        Op::tensor_args_t{.input = input, .gate = gate, .gamma = gamma, .dL_dout = dL_dout});
}

}  // namespace ttnn::prim
