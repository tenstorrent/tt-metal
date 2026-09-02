// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <limits>

#include "subtract_at_target_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

void SubtractAtTargetDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           tt::tt_metal::Layout required_layout,
                           tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "SubtractAtTarget: '{}' must be on DEVICE, got '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "SubtractAtTarget: '{}' buffer is null.", name);
        TT_FATAL(
            tensor.layout() == required_layout,
            "SubtractAtTarget: '{}' must have layout '{}', got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == required_dtype,
            "SubtractAtTarget: '{}' must have dtype '{}', got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "SubtractAtTarget: '{}' must use INTERLEAVED memory layout, got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.input, "input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.target, "target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);

    TT_FATAL(
        tensor_args.input.logical_shape().rank() == 4U,
        "SubtractAtTarget: input must be rank 4, got rank {}",
        tensor_args.input.logical_shape().rank());

    // The reader walks one row-major target page per batch-channel slice of the input
    // (page = tile_row / Ht over NC * Ht rows) and sizes each page read from the target's
    // inner dim, while the program cache is keyed on the input shape alone. Pinning both the
    // target's page width and its page count to the input keeps every page index the reader
    // can form inside the target allocation, and keeps a cached program valid for the target
    // tensor it runs with.
    const auto& target_shape = tensor_args.target.logical_shape();
    TT_FATAL(
        target_shape[-1] == tensor_args.input.logical_shape()[-2],
        "SubtractAtTarget: target inner dim ({}) must equal input sequence dim ({})",
        target_shape[-1],
        tensor_args.input.logical_shape()[-2]);
    const auto& input_padded_shape = tensor_args.input.padded_shape();
    const uint64_t input_nc_pages =
        input_padded_shape.volume() / (static_cast<uint64_t>(input_padded_shape[-2]) * input_padded_shape[-1]);
    const uint64_t target_pages = target_shape.volume() / target_shape[-1];
    TT_FATAL(
        target_pages == input_nc_pages,
        "SubtractAtTarget: target must supply one page per input batch-channel slice, got {} page(s) for {} "
        "slice(s)",
        target_pages,
        input_nc_pages);

    TT_FATAL(args.local_V > 0U, "SubtractAtTarget: local_V must be > 0");

    if (args.cluster_axis.has_value()) {
        auto* device = tensor_args.input.device();
        TT_FATAL(device != nullptr, "SubtractAtTarget: input must be on a (mesh) device");
        const auto mesh_shape = device->shape();
        TT_FATAL(
            *args.cluster_axis < mesh_shape.dims(),
            "SubtractAtTarget: cluster_axis ({}) is out of range for mesh shape with {} dim(s)",
            *args.cluster_axis,
            mesh_shape.dims());
    }

    if (tensor_args.preallocated_output.has_value()) {
        check_tensor(
            tensor_args.preallocated_output.value(),
            "preallocated_output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);
    }
}

SubtractAtTargetDeviceOperation::spec_return_value_t SubtractAtTargetDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    return tensor_args.input.tensor_spec();
}

SubtractAtTargetDeviceOperation::tensor_return_value_t SubtractAtTargetDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t SubtractAtTargetDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // first_v / local_V / subtract_value only affect runtime args (they're patched by
    // override_runtime_arguments per coord). cluster_axis, however, determines the mesh-workload
    // structure (one program per TP slab when set vs one per coordinate when unset) and the
    // program-to-coordinate mapping, so it must be part of the hash. value_or keeps nullopt
    // distinct from axis 0 (an optional hashes its payload directly, so nullopt and 0 would
    // otherwise collide); the sentinel can never be a valid axis.
    return tt::tt_metal::operation::hash_operation<SubtractAtTargetDeviceOperation>(
        args.cluster_axis.value_or(std::numeric_limits<uint32_t>::max()),
        tensor_args.input.dtype(),
        tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::subtract_at_target::device

namespace ttnn::prim {

ttml::metal::ops::subtract_at_target::device::SubtractAtTargetDeviceOperation::tensor_return_value_t
ttml_subtract_at_target(
    const ttnn::Tensor& input,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v,
    const std::optional<ttnn::Tensor>& preallocated_output,
    float subtract_value) {
    using OperationType = ttml::metal::ops::subtract_at_target::device::SubtractAtTargetDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .first_v = first_v, .local_V = local_V, .cluster_axis = cluster_axis, .subtract_value = subtract_value};
    auto tensor_args =
        OperationType::tensor_args_t{.input = input, .target = target, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
