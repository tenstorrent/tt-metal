// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

namespace {

void check_tensor(const char* op_name, const ttnn::Tensor& tensor, const char* name, const ttnn::Tensor& reference) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
        "{}: {} must be allocated on device",
        op_name,
        name);
    TT_FATAL(tensor.device() == reference.device(), "{}: {} must be on the input's device", op_name, name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "{}: {} must be TILE, got {}",
        op_name,
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "{}: {} must be BFLOAT16, got {}",
        op_name,
        name,
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{}: {} must be DRAM interleaved",
        op_name,
        name);
}

}  // namespace

GatedRmsNormGeometry validate_and_get_geometry(
    const char* op_name,
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const std::optional<ttnn::Tensor>& dL_dout) {
    check_tensor(op_name, input, "input", input);
    check_tensor(op_name, gate, "gate", input);
    check_tensor(op_name, gamma, "gamma", input);
    if (dL_dout.has_value()) {
        check_tensor(op_name, *dL_dout, "dL_dout", input);
    }

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[1] == 1, "{}: input must be [B, 1, T, W], got {}", op_name, shape);
    TT_FATAL(gate.logical_shape() == shape, "{}: gate must match the input shape", op_name);
    if (dL_dout.has_value()) {
        TT_FATAL(dL_dout->logical_shape() == shape, "{}: dL_dout must match the input shape", op_name);
    }

    const auto& gshape = gamma.logical_shape();
    const uint32_t group = gshape[-1];
    TT_FATAL(gamma.logical_volume() == group, "{}: gamma must be [1, 1, 1, group], got {}", op_name, gshape);
    const uint32_t seq = shape[-2];
    const uint32_t width = shape[-1];
    TT_FATAL(
        seq % tt::constants::TILE_HEIGHT == 0,
        "{}: T must be a multiple of {}, got {}",
        op_name,
        tt::constants::TILE_HEIGHT,
        seq);
    TT_FATAL(
        group > 0 && group % tt::constants::TILE_WIDTH == 0,
        "{}: group (gamma width) must be a positive multiple of {}, got {}",
        op_name,
        tt::constants::TILE_WIDTH,
        group);
    TT_FATAL(width % group == 0, "{}: W = {} must be a multiple of group = {}", op_name, width, group);

    GatedRmsNormGeometry geo;
    geo.rows_tiles = (shape[0] * seq) / tt::constants::TILE_HEIGHT;
    geo.width_tiles = width / tt::constants::TILE_WIDTH;
    geo.group_tiles = group / tt::constants::TILE_WIDTH;
    geo.num_groups = width / group;
    geo.group = group;
    return geo;
}

void GatedRmsNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    (void)validate_and_get_geometry(
        "gated_rmsnorm_fw", tensor_args.input, tensor_args.gate, tensor_args.gamma, std::nullopt);
}

fw::spec_return_value_t GatedRmsNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.input.memory_config()));
}

fw::tensor_return_value_t GatedRmsNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t GatedRmsNormForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<GatedRmsNormForwardDeviceOperation>(
        args.epsilon, tensor_args.input.logical_shape(), tensor_args.gamma.logical_shape());
}

}  // namespace ttml::metal::ops::gated_rmsnorm::device

namespace ttnn::prim {

ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormForwardDeviceOperation::tensor_return_value_t
ttml_gated_rmsnorm_fw(const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, float epsilon) {
    using OperationType = ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormForwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{.epsilon = epsilon};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .gate = gate, .gamma = gamma};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
