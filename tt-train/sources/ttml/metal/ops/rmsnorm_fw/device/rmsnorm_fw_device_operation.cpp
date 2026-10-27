// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw_device_operation.hpp"

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "metal/ops/common/rmsnorm_common.hpp"
#include "rmsnorm_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

using rmsnorm::get_output_shape;
using rmsnorm::get_rms_shape;

void RMSNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    constexpr std::string_view op = "RMSNormForward";
    // The kernels address DRAM banks directly. Outputs the op allocates take the input's memory config.
    const DeviceTensorRequirements in_dram{.buffer_type = tt::tt_metal::BufferType::DRAM};
    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& preallocated_rms_tensor = tensor_args.preallocated_rms;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_device_tensor(
        input_tensor,
        op,
        "Input",
        {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32},
         .buffer_type = tt::tt_metal::BufferType::DRAM});
    check_device_tensor(gamma_tensor, op, "Gamma", in_dram);
    rmsnorm::check_input_and_gamma_shapes(input_tensor, gamma_tensor, op);
    if (tensor_args.gate.has_value()) {
        check_device_tensor(tensor_args.gate.value(), op, "Gate", in_dram);
        check_head_merged(input_tensor, tensor_args.gate.value(), op, "Gate");
    }
    if (preallocated_rms_tensor.has_value()) {
        check_device_tensor(preallocated_rms_tensor.value(), op, "Preallocated RMS", in_dram);
        check_shape(preallocated_rms_tensor.value(), op, "Preallocated RMS", get_rms_shape(input_tensor));
    }
    if (preallocated_output_tensor.has_value()) {
        check_device_tensor(preallocated_output_tensor.value(), op, "Preallocated Output", in_dram);
        check_shape(
            preallocated_output_tensor.value(),
            op,
            "Preallocated Output",
            get_output_shape(input_tensor, tensor_args.gate));
    }
}

spec_return_value_t RMSNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    if (tensor_args.preallocated_output.has_value()) {
        output_specs.push_back(tensor_args.preallocated_output->tensor_spec());
    } else {
        output_specs.emplace_back(
            get_output_shape(tensor_args.input, tensor_args.gate),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    if (args.return_intermediates) {
        if (tensor_args.preallocated_rms.has_value()) {
            output_specs.push_back(tensor_args.preallocated_rms->tensor_spec());
        } else {
            output_specs.emplace_back(
                get_rms_shape(tensor_args.input),
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
        }
    }

    return output_specs;
}

tensor_return_value_t RMSNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_output.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    if (args.return_intermediates) {
        if (tensor_args.preallocated_rms.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_rms.value());
        } else {
            output_tensors.push_back(ttnn::create_device_tensor(output_specs[1], tensor_args.input.device()));
        }
    }

    return output_tensors;
}

ttsl::hash::hash_t RMSNormForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    // 0 is the ungated mode, which differs from gated mode with a single head.
    const uint32_t num_heads =
        tensor_args.gate.has_value() ? get_num_heads_of_head_merged(input_tensor, tensor_args.gate.value()) : 0U;
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RMSNormForwardDeviceOperation>(
        args, input_tensor.dtype(), input_logical_shape, num_heads);

    return hash;
}

}  // namespace ttml::metal::ops::rmsnorm_fw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const std::optional<ttnn::Tensor>& gate,
    bool return_intermediates,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_rms,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.return_intermediates = return_intermediates, .epsilon = epsilon};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .gate = gate,
        .preallocated_rms = preallocated_rms,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
