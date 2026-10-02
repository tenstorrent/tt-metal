// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "cross_entropy_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

namespace {

tt::tt_metal::TensorSpec expected_output_spec(const ttnn::Tensor& input) {
    return input.tensor_spec();
}

void validate_tensor(
    const ttnn::Tensor& tensor,
    const std::string& name,
    tt::tt_metal::Layout required_layout,
    tt::tt_metal::DataType required_dtype,
    bool require_canonical_spec) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "CrossEntropyBackward operation requires '{}' to be on DEVICE. Got storage type: '{}'",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);
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
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Tensor '{}' must use INTERLEAVED memory layout, but got '{}'",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Tensor '{}' must be in DRAM, but got '{}'",
        name,
        enchantum::to_string(tensor.buffer()->buffer_type()));

    if (require_canonical_spec) {
        const auto canonical_spec = tt::tt_metal::TensorSpec(
            tensor.logical_shape(),
            tt::tt_metal::TensorLayout(required_dtype, required_layout, tensor.memory_config()));
        TT_FATAL(
            tensor.tensor_spec() == canonical_spec,
            "Tensor '{}' must use canonical physical shape and page geometry for its logical shape",
            name);
    }
}

}  // namespace

void CrossEntropyBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& target_tensor = tensor_args.target;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    validate_tensor(input_tensor, "Input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16, true);
    validate_tensor(target_tensor, "Target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32, false);

    auto* device = input_tensor.device();
    TT_FATAL(device != nullptr, "CrossEntropyBackward: input must be on a (mesh) device");
    TT_FATAL(target_tensor.device() == device, "CrossEntropyBackward: target must be on the same device as input");
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4U,
        "CrossEntropyBackward: input must be rank 4, got rank {}",
        input_tensor.logical_shape().rank());
    TT_FATAL(
        target_tensor.logical_shape().rank() >= 1U, "CrossEntropyBackward: target must have at least one dimension");

    // The reader walks one row-major target page per batch-channel slice of the input
    // (page = tile_row / Ht over NC * Ht rows) and sizes each page read from the target's
    // inner dim, while the program cache is keyed on the input shape alone. Pinning both the
    // target's page width and its page count to the input keeps every page index the reader
    // can form inside the target allocation, and keeps a cached program valid for the target
    // tensor it runs with.
    const auto& target_shape = target_tensor.logical_shape();
    TT_FATAL(
        target_shape[-1] == input_tensor.logical_shape()[-2],
        "CrossEntropyBackward: target inner dim ({}) must equal input sequence dim ({})",
        target_shape[-1],
        input_tensor.logical_shape()[-2]);
    const auto& input_padded_shape = input_tensor.padded_shape();
    const uint64_t input_nc_pages =
        input_padded_shape.volume() / (static_cast<uint64_t>(input_padded_shape[-2]) * input_padded_shape[-1]);
    const uint64_t target_pages = target_shape.volume() / target_shape[-1];
    TT_FATAL(
        target_pages == input_nc_pages,
        "CrossEntropyBackward: target must supply one page per input batch-channel slice, got {} page(s) for {} "
        "slice(s)",
        target_pages,
        input_nc_pages);

    if (preallocated_output_tensor.has_value()) {
        const auto& output = preallocated_output_tensor.value();
        validate_tensor(
            output, "Preallocated Output", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16, true);
        TT_FATAL(output.device() == device, "CrossEntropyBackward: output must be on the same device as input");
        TT_FATAL(
            output.tensor_spec() == expected_output_spec(input_tensor),
            "CrossEntropyBackward: preallocated output TensorSpec must exactly match the derived output TensorSpec");
    }
}

CrossEntropyBackwardDeviceOperation::spec_return_value_t CrossEntropyBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return expected_output_spec(tensor_args.input);
}

CrossEntropyBackwardDeviceOperation::tensor_return_value_t CrossEntropyBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensor = tensor_args.preallocated_output.value();
    } else {
        output_tensor = ttnn::create_device_tensor(output_specs, tensor_args.input.device());
    }

    return output_tensor;
}

ttsl::hash::hash_t CrossEntropyBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    return tt::tt_metal::operation::hash_operation<CrossEntropyBackwardDeviceOperation>(
        args, input_tensor.dtype(), input_logical_shape);
}

}  // namespace ttml::metal::ops::cross_entropy_bw::device

namespace ttnn::prim {

ttml::metal::ops::cross_entropy_bw::device::CrossEntropyBackwardDeviceOperation::tensor_return_value_t
ttml_cross_entropy_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& target_tensor,
    float scaler,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::cross_entropy_bw::device::CrossEntropyBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.scaler = scaler};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .target = target_tensor,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
