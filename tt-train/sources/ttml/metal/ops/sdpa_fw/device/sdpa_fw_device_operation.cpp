// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_device_operation.hpp"

#include "metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp"
#include "sdpa_fw_program_factory.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

SDPAForwardDeviceOperation::program_factory_t SDPAForwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return SDPAForwardProgramFactory{};
}

void SDPAForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SDPAForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& query = tensor_args.query;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const auto& mask = tensor_args.mask;

    // as I understand I should use TT_FATAL here instead of throwing exceptions
    // check rank(rank must be 4)
    // check shapes(same for all)
    // check heads/groups
    // check tensor data type, layout, memory layout, storage type, device arch
    // check mask???
}

spec_return_value_t SDPAForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    if (tensor_args.preallocated_output.has_value()) {
        output_specs.push_back(tensor_args.preallocated_output->tensor_spec());
    } else {
        auto shape = tensor_args.query.logical_shape();  // output shape is the same as query shape
        // shape[3] = shape[2];
        output_specs.emplace_back(
            shape,
            tt::tt_metal::TensorLayout(
                tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    }

    if (args.return_intermediates) {
        // TODO: add intermediate spec when I'll know what exactly I want to return
    }

    return output_specs;
}

tensor_return_value_t SDPAForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_output.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.query.device()));
    }

    if (args.return_intermediates) {
        // TODO: add intermediate tensor when I'll know what exactly I want to return
    }

    return output_tensors;
}

ttsl::hash::hash_t SDPAForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // TODO[change]: calculation of hash could be changed due to sdpa implementation
    // query shape should  difine the shape of other inputs and outputs
    // we assume that query, key and value have the same shape, and we validate it in validate function
    const auto& query_tensor = tensor_args.query;
    const auto& query_logical_shape = query_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<SDPAForwardDeviceOperation>(
        args, program_factory.index(), query_tensor.dtype(), query_logical_shape);

    return hash;
}

std::tuple<SDPAForwardDeviceOperation::operation_attributes_t, SDPAForwardDeviceOperation::tensor_args_t>
SDPAForwardDeviceOperation::invoke(
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& key_tensor,
    const ttnn::Tensor& value_tensor,
    const std::optional<ttnn::Tensor>& mask,  // attention mask
    float dropout_probability,                // default value
    bool return_intermediates,
    const std::optional<ttnn::Tensor>& preallocated_intermediate,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    operation_attributes_t operation_attributes{
        .return_intermediates = return_intermediates, .dropout_probability = dropout_probability};
    tensor_args_t tensor_args{
        .query = query_tensor,
        .key = key_tensor,
        .value = value_tensor,
        .mask = mask,
        .preallocated_intermediate = preallocated_intermediate,
        .preallocated_output = preallocated_output,
    };

    return {operation_attributes, tensor_args};
}

}  // namespace ttml::metal::ops::sdpa_fw::device
