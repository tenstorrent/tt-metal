// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target_device_operation.hpp"

#include <limits>

#include "metal/common/tensor_validation.hpp"
#include "subtract_at_target_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

void SubtractAtTargetDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_device_tensor(tensor_args.input, "SubtractAtTarget", "input");
    check_device_tensor(
        tensor_args.target,
        "SubtractAtTarget",
        "target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});

    TT_FATAL(
        tensor_args.input.logical_shape().rank() == 4U,
        "SubtractAtTarget: input must be rank 4, got rank {}",
        tensor_args.input.logical_shape().rank());

    // The reader walks one row-major target page per batch-channel slice of the input
    // (page = tile_row / Ht over NC * Ht rows) and sizes each page read from the target's
    // inner dim. Pinning both the target's page width and its page count to the input keeps
    // every page index the reader can form inside the target allocation.
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
        check_device_tensor(tensor_args.preallocated_output.value(), "SubtractAtTarget", "preallocated_output");
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

// first_v, local_V and subtract_value are left out of the key: they reach the kernels only through the device's shard
// window and the subtracted value, runtime args that SubtractAtTargetProgramFactory::override_runtime_arguments
// re-applies on every cache hit, so e.g. MEAN (1/N) and NONE (1.0) share one program. cluster_axis and the full spec of
// every tensor, including whether a preallocated output is passed, are keyed. value_or keeps nullopt distinct from
// axis 0 (libc++'s std::hash maps an empty optional and a 0 payload to the same value); the sentinel can never be a
// valid axis.
ttsl::hash::hash_t SubtractAtTargetDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SubtractAtTargetDeviceOperation>(
        args.cluster_axis.value_or(std::numeric_limits<uint32_t>::max()), tensor_args);
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
