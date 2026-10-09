// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

namespace {

constexpr uint32_t kTile = tt::constants::TILE_WIDTH;

void check_tensor(const ttnn::Tensor& tensor, std::string_view name, std::string_view op) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "{}: {} must be on device, got storage type {}",
        op,
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "{}: {} must be allocated on device", op, name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "{}: {} must be in TILE layout, got {}",
        op,
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "{}: {} must be BFLOAT16, got {}",
        op,
        name,
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            tensor.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{}: {} must be DRAM interleaved, got {}",
        op,
        name,
        tensor.memory_config());
}

void check_same_shape_and_device(
    const ttnn::Tensor& tensor, const ttnn::Tensor& input, std::string_view name, std::string_view op) {
    TT_FATAL(
        tensor.logical_shape() == input.logical_shape(),
        "{}: {} shape {} must equal input shape {}",
        op,
        name,
        tensor.logical_shape(),
        input.logical_shape());
    TT_FATAL(tensor.device() == input.device(), "{}: {} must be on the same device as input", op, name);
}

}  // namespace

GatedRmsNormGeometry validate_and_get_geometry(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const std::optional<ttnn::Tensor>& dL_dout,
    std::string_view op_name) {
    check_tensor(input, "input", op_name);
    check_tensor(gate, "gate", op_name);
    check_tensor(gamma, "gamma", op_name);
    check_same_shape_and_device(gate, input, "gate", op_name);
    TT_FATAL(gamma.device() == input.device(), "{}: gamma must be on the same device as input", op_name);
    if (dL_dout.has_value()) {
        check_tensor(*dL_dout, "dL_dout", op_name);
        check_same_shape_and_device(*dL_dout, input, "dL_dout", op_name);
    }

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4U && shape[1] == 1U, "{}: input must be [B, 1, T, W], got {}", op_name, shape);
    const uint32_t T = shape[2];
    const uint32_t W = shape[3];
    TT_FATAL(T % kTile == 0U, "{}: T = {} must be a multiple of {}", op_name, T, kTile);

    const auto& gamma_shape = gamma.logical_shape();
    TT_FATAL(
        gamma_shape.rank() == 4U && gamma_shape[0] == 1U && gamma_shape[1] == 1U && gamma_shape[2] == 1U,
        "{}: gamma must be [1, 1, 1, V], got {}",
        op_name,
        gamma_shape);
    const uint32_t V = gamma_shape[3];
    TT_FATAL(V > 0U && V % kTile == 0U, "{}: V = {} must be a positive multiple of {}", op_name, V, kTile);
    TT_FATAL(W % V == 0U, "{}: input width W = {} must be a multiple of V = {}", op_name, W, V);

    return GatedRmsNormGeometry{
        .rows_tiles = shape[0] * T / kTile,
        .width_tiles = W / kTile,
        .group_tiles = V / kTile,
        .num_groups = W / V,
        .group = V,
    };
}

void GatedRmsNormFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    validate_and_get_geometry(tensor_args.input, tensor_args.gate, tensor_args.gamma, std::nullopt, "GatedRmsNormFw");
}

GatedRmsNormFwDeviceOperation::spec_return_value_t GatedRmsNormFwDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, input.memory_config()));
}

GatedRmsNormFwDeviceOperation::tensor_return_value_t GatedRmsNormFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t GatedRmsNormFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<GatedRmsNormFwDeviceOperation>(
        args, tensor_args.input.logical_shape(), tensor_args.gamma.logical_shape());
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device

namespace ttnn::prim {

ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormFwDeviceOperation::tensor_return_value_t ttml_gated_rmsnorm_fw(
    const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, const float epsilon) {
    using Op = ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormFwDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{.epsilon = epsilon},
        Op::tensor_args_t{.input = input, .gate = gate, .gamma = gamma});
}

}  // namespace ttnn::prim
