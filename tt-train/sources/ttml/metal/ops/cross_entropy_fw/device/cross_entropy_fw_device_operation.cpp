// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "cross_entropy_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

void CrossEntropyForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype,
                           bool allow_sharded = false) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "CrossEntropyForward operation requires '{}' to be on DEVICE. Got storage type: '{}'",
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

        auto memory_layout = tensor.memory_config().memory_layout();
        bool is_valid_layout = (memory_layout == ttnn::TensorMemoryLayout::INTERLEAVED) ||
                               (allow_sharded && memory_layout == ttnn::TensorMemoryLayout::HEIGHT_SHARDED) ||
                               (allow_sharded && memory_layout == ttnn::TensorMemoryLayout::BLOCK_SHARDED);
        TT_FATAL(
            is_valid_layout,
            "Tensor '{}' must use INTERLEAVED{} memory layout, but got '{}'",
            name,
            allow_sharded ? " or HEIGHT_SHARDED or BLOCK_SHARDED" : "",
            enchantum::to_string(memory_layout));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& target_tensor = tensor_args.target;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    // Input tensor can be sharded (HEIGHT_SHARDED or BLOCK_SHARDED via NdShardSpec) for DRAM sharding optimization
    check_tensor(
        input_tensor, "Input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16, /*allow_sharded=*/true);
    check_tensor(target_tensor, "Target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);
    if (preallocated_output_tensor.has_value()) {
        check_tensor(
            preallocated_output_tensor.value(),
            "Preallocated Output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);
    }
}

CrossEntropyForwardDeviceOperation::spec_return_value_t CrossEntropyForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    auto input_logical_shape = tensor_args.input.logical_shape();
    input_logical_shape[-1] = 1U;

    // Output tensor has different shape than input (width=1), so we can't use
    // the input's sharded memory config directly. Use INTERLEAVED DRAM for output.
    auto output_memory_config = tensor_args.input.memory_config();
    if (output_memory_config.is_sharded()) {
        output_memory_config = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};
    }

    return ttnn::TensorSpec(
        ttnn::Shape(input_logical_shape),
        tt::tt_metal::TensorLayout(tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, output_memory_config));
}

CrossEntropyForwardDeviceOperation::tensor_return_value_t CrossEntropyForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensor = tensor_args.preallocated_output.value();
    } else {
        output_tensor = create_device_tensor(output_specs, tensor_args.input.device());
    }

    return output_tensor;
}

ttsl::hash::hash_t CrossEntropyForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    // Include memory layout in hash so sharded and interleaved tensors use different cached programs
    auto hash = tt::tt_metal::operation::hash_operation<CrossEntropyForwardDeviceOperation>(
        args, input_tensor.dtype(), input_logical_shape, input_tensor.memory_config().memory_layout());

    return hash;
}

}  // namespace ttml::metal::ops::cross_entropy_fw::device

namespace ttnn::prim {

ttml::metal::ops::cross_entropy_fw::device::CrossEntropyForwardDeviceOperation::tensor_return_value_t
ttml_cross_entropy_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& target_tensor,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::cross_entropy_fw::device::CrossEntropyForwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .target = target_tensor,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
