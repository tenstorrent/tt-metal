// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_device_operation.hpp"

#include <algorithm>

#include "tt_stl/assert.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttml::metal::ops::softmax_backward::device {

namespace {

bool is_canonical_tile(const Tile& tile) {
    const auto canonical = Tile{};
    return tile.get_tile_shape() == canonical.get_tile_shape() && tile.get_face_shape() == canonical.get_face_shape() &&
           tile.get_num_faces() == canonical.get_num_faces() && !tile.get_transpose_within_face() &&
           !tile.get_transpose_of_faces();
}

}  // namespace

void SoftmaxBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;
    TT_FATAL(softmax_output.logical_shape().rank() == 4, "Softmax backward requires rank-4 tensors");
    TT_FATAL(
        softmax_output.device() == upstream_grad.device(),
        "Softmax output and upstream gradient must be on the same device");
    TT_FATAL(
        std::ranges::equal(softmax_output.device_storage().get_coords(), upstream_grad.device_storage().get_coords()),
        "Softmax output and upstream gradient must cover the same device coordinates");
    TT_FATAL(
        softmax_output.logical_shape() == upstream_grad.logical_shape(),
        "Softmax output and upstream gradient tensors must have the same shape");
    TT_FATAL(
        softmax_output.padded_shape() == upstream_grad.padded_shape(),
        "Softmax output and upstream gradient tensors must have the same padded shape");
    TT_FATAL(
        softmax_output.dtype() == DataType::BFLOAT16 || softmax_output.dtype() == DataType::FLOAT32,
        "Softmax backward only supports BFLOAT16 and FLOAT32");
    TT_FATAL(
        upstream_grad.dtype() == softmax_output.dtype(),
        "Softmax output and upstream gradient must have the same dtype");
    TT_FATAL(softmax_output.layout() == Layout::TILE, "Softmax backward requires TILE layout");
    TT_FATAL(upstream_grad.layout() == Layout::TILE, "Softmax backward requires TILE layout");
    TT_FATAL(
        softmax_output.tensor_spec().page_config() == upstream_grad.tensor_spec().page_config(),
        "Softmax output and upstream gradient must have the same page configuration");
    TT_FATAL(
        is_canonical_tile(softmax_output.tensor_spec().tile()) && is_canonical_tile(upstream_grad.tensor_spec().tile()),
        "Softmax backward requires the canonical 32x32 tile");
    const auto& logical_shape = softmax_output.logical_shape();
    const auto& padded_shape = softmax_output.padded_shape();
    const auto tile_width = Tile{}.get_width();
    const auto minimum_padded_width = ((logical_shape[-1] + tile_width - 1U) / tile_width) * tile_width;
    TT_FATAL(
        padded_shape[0] == logical_shape[0] && padded_shape[1] == logical_shape[1] &&
            padded_shape[-1] == minimum_padded_width,
        "Softmax backward only supports padding in the height dimension (logical shape {}, padded shape {})",
        logical_shape,
        padded_shape);
    const auto rank = softmax_output.logical_shape().rank();
    TT_FATAL(
        attributes.dim == rank - 1,
        "Currently only supporting softmax_backward on last dimension (got dim={}, rank={})",
        attributes.dim,
        rank);
}

SoftmaxBackwardDeviceOperation::spec_return_value_t SoftmaxBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.softmax_output.tensor_spec();
}

SoftmaxBackwardDeviceOperation::tensor_return_value_t SoftmaxBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return ttnn::create_device_tensor(output_spec, tensor_args.softmax_output.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SoftmaxBackwardDeviceOperation::tensor_return_value_t>
SoftmaxBackwardDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;
    const auto& output_tensor = tensor_return_value;
    // Placeholder; tt-train build does not depend on ttnn data_movement common_tm_bw_model.
    constexpr int ideal_dev_clock_cycles = 0;
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {softmax_output, upstream_grad}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttml::metal::ops::softmax_backward::device

namespace ttnn::prim {

ttnn::Tensor ttml_softmax_backward(
    const ttnn::Tensor& softmax_output,
    const ttnn::Tensor& upstream_grad,
    uint32_t dim,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    using OperationType = ttml::metal::ops::softmax_backward::device::SoftmaxBackwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{dim, sub_core_grids};
    auto tensor_args = OperationType::tensor_args_t{softmax_output, upstream_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
