// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "select_target_logit_device_operation.hpp"

#include <limits>

#include "metal/common/tensor_validation.hpp"
#include "select_target_logit_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::select_target_logit::device {

void SelectTargetLogitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_device_tensor(tensor_args.logit, "SelectTargetLogit", "logit");
    check_device_tensor(
        tensor_args.target,
        "SelectTargetLogit",
        "target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});

    TT_FATAL(
        tensor_args.logit.logical_shape().rank() == 4U,
        "SelectTargetLogit: logit must be rank 4, got rank {}",
        tensor_args.logit.logical_shape().rank());

    // The reader walks one row-major target page per batch-channel slice of the logit
    // (page = tile_row / Ht over NC * Ht rows) and sizes each page read from the target's
    // inner dim, while the program cache is keyed on the logit shape alone. Pinning both the
    // target's page width and its page count to the logit keeps every page index the reader
    // can form inside the target allocation, and keeps a cached program valid for the target
    // tensor it runs with.
    const auto& target_shape = tensor_args.target.logical_shape();
    TT_FATAL(
        target_shape[-1] == tensor_args.logit.logical_shape()[-2],
        "SelectTargetLogit: target inner dim ({}) must equal logit sequence dim ({})",
        target_shape[-1],
        tensor_args.logit.logical_shape()[-2]);
    const auto& logit_padded_shape = tensor_args.logit.padded_shape();
    const uint64_t logit_nc_pages =
        logit_padded_shape.volume() / (static_cast<uint64_t>(logit_padded_shape[-2]) * logit_padded_shape[-1]);
    const uint64_t target_pages = target_shape.volume() / target_shape[-1];
    TT_FATAL(
        target_pages == logit_nc_pages,
        "SelectTargetLogit: target must supply one page per logit batch-channel slice, got {} page(s) for {} "
        "slice(s)",
        target_pages,
        logit_nc_pages);

    TT_FATAL(args.local_V > 0U, "SelectTargetLogit: local_V must be > 0");

    if (args.cluster_axis.has_value()) {
        auto* device = tensor_args.logit.device();
        TT_FATAL(device != nullptr, "SelectTargetLogit: logit must be on a (mesh) device");
        const auto mesh_shape = device->shape();
        TT_FATAL(
            *args.cluster_axis < mesh_shape.dims(),
            "SelectTargetLogit: cluster_axis ({}) is out of range for mesh shape with {} dim(s)",
            *args.cluster_axis,
            mesh_shape.dims());
    }

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        check_device_tensor(out, "SelectTargetLogit", "preallocated_output");
    }
}

SelectTargetLogitDeviceOperation::spec_return_value_t SelectTargetLogitDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    ttnn::Shape shape = tensor_args.logit.logical_shape();
    shape[-1] = 1U;
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tensor_args.logit.dtype(), tt::tt_metal::Layout::TILE, tensor_args.logit.memory_config()));
}

SelectTargetLogitDeviceOperation::tensor_return_value_t SelectTargetLogitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.logit.device());
}

ttsl::hash::hash_t SelectTargetLogitDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // first_v / local_V only affect runtime args (they're patched by override_runtime_arguments
    // per coord). cluster_axis, however, determines the mesh-workload structure (one program per
    // TP slab when set vs one per coordinate when unset) and the program-to-coordinate mapping,
    // so it must be part of the hash. value_or keeps nullopt distinct from axis 0 (an optional
    // hashes its payload directly, so nullopt and 0 would otherwise collide); the sentinel can
    // never be a valid axis.
    return tt::tt_metal::operation::hash_operation<SelectTargetLogitDeviceOperation>(
        args.cluster_axis.value_or(std::numeric_limits<uint32_t>::max()),
        tensor_args.logit.dtype(),
        tensor_args.logit.logical_shape());
}

}  // namespace ttml::metal::ops::select_target_logit::device

namespace ttnn::prim {

ttml::metal::ops::select_target_logit::device::SelectTargetLogitDeviceOperation::tensor_return_value_t
ttml_select_target_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::select_target_logit::device::SelectTargetLogitDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.first_v = first_v, .local_V = local_V, .cluster_axis = cluster_axis};
    auto tensor_args =
        OperationType::tensor_args_t{.logit = logit, .target = target, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
