// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "metal/ops/sdpa_fw/device/sdpa_fw_device_operation_types.hpp"
#include "sdpa_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

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

    auto query_shape = query.logical_shape();
    auto key_shape = key.logical_shape();
    auto value_shape = value.logical_shape();

    auto [qBt, qHt, qSt, qEt] = query_shape.to_array_4D();
    auto [kBt, kHt, kSt, kEt] = key_shape.to_array_4D();
    auto [vBt, vHt, vSt, vEt] = value_shape.to_array_4D();

    TT_FATAL(
        qHt > 0 && kHt > 0 && vHt > 0, "Number of heads must be greater than zero. Got heads={}, groups={}", qHt, kHt);
    TT_FATAL(
        qHt % kHt == 0,
        "Number of query heads ({}) must be divisible by number of key/value heads ({}) for grouped attention. "
        "This ensures each key/value group serves an integer number of query heads.",
        qHt,
        kHt);

    TT_FATAL(
        qBt == kBt && qBt == vBt && qSt == kSt && qSt == vSt && qEt == kEt && qEt == vEt,
        "Query, Key and Value must have the same batch size and sequence length, except for  number of heads. Got "
        "shapes: Query={}, Key={}, "
        "Value={}",
        query_shape,
        key_shape,
        value_shape);

    TT_FATAL(
        key_shape == value_shape,
        "Key and Value must have the same shape. Got Key={}, Value={}",
        key_shape,
        value_shape);

    // Validate mask shape if provided - must be (1, 1, S, S)
    if (tensor_args.mask.has_value()) {
        const auto& mask = tensor_args.mask.value();
        check_tensor(mask, "Attention Mask", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);

        auto mask_shape = mask.logical_shape();
        auto [mB, mH, mS1, mS2] = mask_shape.to_array_4D();

        TT_FATAL(
            mB == 1 && mH == 1,
            "Attention mask must have shape (1, 1, S, S) for broadcasting. "
            "Got mask shape ({}, {}, {}, {}). "
            "Full (B, H, S, S) masks will be supported in a future PR.",
            mB,
            mH,
            mS1,
            mS2);

        TT_FATAL(
            mS1 == qSt && mS2 == qSt,
            "Attention mask sequence dimensions must match query sequence length. "
            "Got mask shape ({}, {}, {}, {}), expected (1, 1, {}, {})",
            mB,
            mH,
            mS1,
            mS2,
            qSt,
            qSt);
    }

    if (preallocated_output.has_value()) {
        check_tensor(
            preallocated_output.value(),
            "Preallocated Output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);

        auto output_shape = preallocated_output->padded_shape();
        // Output shape (B, H, S, D) - heads NOT fused
        TT_FATAL(
            output_shape[0] == query_shape[0] &&      // B
                output_shape[1] == qHt &&             // H (heads NOT fused)
                output_shape[2] == query_shape[2] &&  // S
                output_shape[3] == query_shape[3],    // D
            "Invalid preallocated output shape. Expected (B, H, S, D) = ({}, {}, {}, {}), got {}. Query shape={}",
            query_shape[0],
            qHt,
            query_shape[2],
            query_shape[3],
            output_shape,
            query_shape);
    }

    // TODO(vmelnykov): #28205 - Implement dropout support in SDPA forward operation
    // Currently dropout is not implemented
    TT_FATAL(
        args.dropout_probability == 0.0F,
        "Dropout is not currently supported in SDPA forward operation. "
        "Got dropout_probability={}, expected 0.0. "
        "See ticket #28205 for implementation status.",
        args.dropout_probability);

    if (args.return_intermediates && tensor_args.preallocated_intermediate.has_value()) {
        const auto& preallocated_intermediate = tensor_args.preallocated_intermediate.value();
        check_tensor(
            preallocated_intermediate,
            "Preallocated Intermediate",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);

        auto interm_shape = preallocated_intermediate.padded_shape();
        // intermediate shape: (B, q_heads, S, 64) - max_val at col 0, recip_sum_exp at col 32
        constexpr uint32_t kIntermediateWidth = 64U;
        TT_FATAL(
            interm_shape[0] == qBt && interm_shape[1] == qHt && interm_shape[2] == qSt &&
                interm_shape[3] == kIntermediateWidth,
            "Preallocated intermediate shape must be (B, q_heads, S, 64). Got preallocated intermediate shape={}, "
            "q_heads={}",
            interm_shape,
            qHt);
    }
}

spec_return_value_t SDPAForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    if (tensor_args.preallocated_output.has_value()) {
        output_specs.push_back(tensor_args.preallocated_output->tensor_spec());
    } else {
        auto shape = tensor_args.query.logical_shape();  // output shape is the same as query shape
        // change shape to (B, qNH, S, qEmbd) to match our layout
        // need to fuse heads into last dim in future to avoid extra reshapes and transposes
        // shape[3] = shape[3] * shape[1];  // fused heads in last dim
        // shape[1] = 1U;
        output_specs.emplace_back(
            shape,
            tt::tt_metal::TensorLayout(
                tensor_args.query.dtype(), tt::tt_metal::Layout::TILE, tensor_args.query.memory_config()));
    }

    if (args.return_intermediates) {
        if (tensor_args.preallocated_intermediate.has_value()) {
            output_specs.push_back(tensor_args.preallocated_intermediate->tensor_spec());
        } else {
            auto shape = tensor_args.query.logical_shape();
            // intermediate shape: (B, q_heads, S, 64) - max_val at col 0, recip_sum_exp at col 32
            shape[-1] = 64U;
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
    const auto& query_tensor = tensor_args.query;
    const auto& query_logical_shape = query_tensor.logical_shape();
    const auto& key_tensor = tensor_args.key;
    const auto& key_logical_shape = key_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<SDPAForwardDeviceOperation>(
        args, program_factory.index(), query_tensor.dtype(), query_logical_shape, key_logical_shape);

    return hash;
}

}  // namespace ttml::metal::ops::sdpa_fw::device

namespace ttnn::prim {

ttml::metal::ops::sdpa_fw::device::SDPAForwardDeviceOperation::tensor_return_value_t ttml_sdpa_fw(
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& key_tensor,
    const ttnn::Tensor& value_tensor,
    const std::optional<ttnn::Tensor>& mask,
    const float dropout_probability,
    const bool return_intermediates,
    const std::optional<ttnn::Tensor>& preallocated_intermediate,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::sdpa_fw::device::SDPAForwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .return_intermediates = return_intermediates, .dropout_probability = dropout_probability};
    auto tensor_args = OperationType::tensor_args_t{
        .query = query_tensor,
        .key = key_tensor,
        .value = value_tensor,
        .mask = mask,
        .preallocated_intermediate = preallocated_intermediate,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
