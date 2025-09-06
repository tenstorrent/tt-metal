// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

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
    const auto& preallocated_output = tensor_args.preallocated_output;

    const auto check_tensor = [](const ttnn::Tensor& tensor,
                                 const std::string& name,
                                 const tt::tt_metal::Layout required_layout,
                                 const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SDPAForward operation requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.padded_shape().rank() == 4U,
            "Tensor '{}' must have rank 4, but got rank {}",
            name,
            tensor.padded_shape().rank());

        TT_FATAL(
            tensor.layout() == required_layout,
            "Tensor '{}' must have layout '{}', but got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == required_dtype,
            "Tensor '{}' must have data type '{}', but got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use INTERLEAVED memory layout, but got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };
    check_tensor(query, "Query", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(key, "Key", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(value, "Value", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);

    // TODO[improve]: add check for mask tensor

    auto query_shape = query.padded_shape();
    auto key_shape = key.padded_shape();
    auto value_shape = value.padded_shape();

    const uint32_t q_heads = args.q_heads;    // will be passed by user into args
    const uint32_t kv_heads = args.kv_heads;  // will be passed by user into args
    TT_FATAL(
        q_heads % kv_heads == 0,
        "Number of heads must be divisible by number of groups, got heads={}, groups={}",
        q_heads,
        kv_heads);

    auto [qBt, qHt, qSt, qEt] = query_shape.to_array_4D();
    TT_FATAL(qEt % q_heads == 0, "Query embedding dim must be divisible by number of heads");

    auto [kBt, kHt, kSt, kEt] = key_shape.to_array_4D();
    TT_FATAL(kEt % kv_heads == 0, "Key embedding dim must be divisible by number of key/value groups");

    auto [vBt, vHt, vSt, vEt] = value_shape.to_array_4D();
    TT_FATAL(vEt % kv_heads == 0, "Value embedding dim must be divisible by number of key/value groups");

    TT_FATAL(
        qBt == kBt && qBt == vBt && qSt == kSt && qSt == vSt && qHt == 1U && kHt == 1U && vHt == 1U,
        "Query and Key must have the same shape, except for the inner dim. Got shapes: Query={}, Key={}, Value={}",
        query_shape,
        key_shape,
        value_shape);

    if (preallocated_output.has_value()) {
        check_tensor(
            preallocated_output.value(),
            "Preallocated Output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);

        auto output_shape = preallocated_output->padded_shape();
        TT_FATAL(
            output_shape == query_shape,
            "Preallocated output shape must be the same as query shape. Got preallocated output shape={}, query "
            "shape={}",
            output_shape,
            query_shape);
    }

    // TODO[improve]: add check for intermediate tensor when I'll know what exactly I want to return
}

spec_return_value_t SDPAForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    if (tensor_args.preallocated_output.has_value()) {
        output_specs.push_back(tensor_args.preallocated_output->tensor_spec());
    } else {
        auto shape = tensor_args.query.logical_shape();  // output shape is the same as query shape
        output_specs.emplace_back(
            shape,
            tt::tt_metal::TensorLayout(
                tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    }

    if (args.return_intermediates) {
        // TODO: add intermediate spec when I'll know what exactly I want to return
        if (tensor_args.preallocated_intermediate.has_value()) {
            output_specs.push_back(tensor_args.preallocated_intermediate->tensor_spec());
        } else {
            auto shape = tensor_args.query.logical_shape();
            shape[-1] = 1U;  // intermediate is a scalar per row
            output_specs.emplace_back(
                shape,
                tt::tt_metal::TensorLayout(
                    tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
        }
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
        if (tensor_args.preallocated_intermediate.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_intermediate.value());
        } else {
            output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.query.device()));
        }
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
    const uint32_t q_heads,                   // num of query heads
    const uint32_t kv_heads,                  // num of key/value heads
    const float dropout_probability,          // default value
    const bool return_intermediates,
    const std::optional<ttnn::Tensor>& preallocated_intermediate,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    operation_attributes_t operation_attributes{
        .q_heads = q_heads,
        .kv_heads = kv_heads,
        .return_intermediates = return_intermediates,
        .dropout_probability = dropout_probability};
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
