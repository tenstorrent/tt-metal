// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_allcore_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device {

namespace {

void check_tensor(const ttnn::Tensor& tensor, const std::string& name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "AddRMSNormForward requires {} to be on Device. Input storage type: {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "AddRMSNormForward: {} has no device buffer", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "AddRMSNormForward requires {} in Tile layout, got {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == dtype,
        "AddRMSNormForward requires {} of {} data type, got {}",
        name,
        enchantum::to_string(dtype),
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "AddRMSNormForward requires interleaved {}, got `{}`",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "AddRMSNormForward requires {} in DRAM",
        name);
}

void check_activation(const ttnn::Tensor& t, const std::string& name) {
    check_tensor(t, name, tt::tt_metal::DataType::BFLOAT16);
    const auto& shape = t.logical_shape();
    TT_FATAL(shape.rank() == 4, "{} must be 4D [B, N, S, C], got {}", name, shape);
    TT_FATAL(
        shape[3] % tt::constants::TILE_WIDTH == 0,
        "AddRMSNormForward: C must be a multiple of 32 (no column mask in the kernels), got {}",
        shape[3]);
}

void check_split(uint32_t num_slices, uint32_t slice_tiles, uint32_t Wt) {
    TT_FATAL(num_slices >= 1 && slice_tiles >= 1, "add_rmsnorm_fw: invalid work split");
    TT_FATAL(
        (num_slices - 1U) * slice_tiles < Wt && num_slices * slice_tiles >= Wt,
        "add_rmsnorm_fw: work split ({} slices x {} tiles) does not cover Wt={}",
        num_slices,
        slice_tiles,
        Wt);
}

uint32_t rows_tiles(const ttnn::Tensor& t) {
    const auto& padded = t.padded_shape();
    return padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
}

tt::tt_metal::TensorSpec like(const ttnn::Tensor& t) {
    return tt::tt_metal::TensorSpec(
        t.logical_shape(),
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, t.memory_config()));
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Phase A
// ---------------------------------------------------------------------------------------------------------------

void AddRMSNormForwardPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_activation(tensor_args.residual, "residual");
    check_activation(tensor_args.delta, "delta");
    TT_FATAL(
        tensor_args.delta.logical_shape() == tensor_args.residual.logical_shape(),
        "AddRMSNormForward: residual {} and delta {} must have the same shape",
        tensor_args.residual.logical_shape(),
        tensor_args.delta.logical_shape());
    check_split(args.num_slices, args.slice_tiles, tensor_args.residual.padded_shape()[-1] / tt::constants::TILE_WIDTH);
}

partial::spec_return_value_t AddRMSNormForwardPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t rows = rows_tiles(tensor_args.residual);
    return {
        like(tensor_args.residual),
        tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 1, rows * tt::constants::TILE_HEIGHT, args.num_slices * tt::constants::TILE_WIDTH},
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tensor_args.residual.memory_config()))};
}

partial::tensor_return_value_t AddRMSNormForwardPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.residual.device()));
    }
    return outputs;
}

ttsl::hash::hash_t AddRMSNormForwardPartialDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<AddRMSNormForwardPartialDeviceOperation>(
        args.num_slices, args.slice_tiles, tensor_args.residual.logical_shape());
}

// ---------------------------------------------------------------------------------------------------------------
// Phase B
// ---------------------------------------------------------------------------------------------------------------

void AddRMSNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_activation(tensor_args.sum, "sum");
    check_tensor(tensor_args.gamma, "gamma", tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.partials, "partials", tt::tt_metal::DataType::FLOAT32);
    const auto& shape = tensor_args.sum.logical_shape();
    const auto& gamma_shape = tensor_args.gamma.logical_shape();
    TT_FATAL(
        gamma_shape.rank() == 4 && gamma_shape[0] == 1 && gamma_shape[1] == 1 && gamma_shape[2] == 1 &&
            gamma_shape[3] == shape[3],
        "AddRMSNormForward: gamma must be [1, 1, 1, C={}], got {}",
        shape[3],
        gamma_shape);
    const uint32_t Wt = tensor_args.sum.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    check_split(args.num_slices, args.slice_tiles, Wt);
    const uint32_t rows = rows_tiles(tensor_args.sum);
    const auto& pshape = tensor_args.partials.logical_shape();
    TT_FATAL(
        pshape.rank() == 4 && pshape[2] == rows * tt::constants::TILE_HEIGHT &&
            pshape[3] == args.num_slices * tt::constants::TILE_WIDTH,
        "AddRMSNormForward: partials must be [1, 1, {}, {}], got {}",
        rows * tt::constants::TILE_HEIGHT,
        args.num_slices * tt::constants::TILE_WIDTH,
        pshape);
}

spec_return_value_t AddRMSNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& shape = tensor_args.sum.logical_shape();
    return {
        like(tensor_args.sum),
        tt::tt_metal::TensorSpec(
            ttnn::Shape{shape[0], shape[1], shape[2], 1},
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tensor_args.sum.memory_config()))};
}

tensor_return_value_t AddRMSNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.sum.device()));
    }
    return outputs;
}

ttsl::hash::hash_t AddRMSNormForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<AddRMSNormForwardDeviceOperation>(
        args.epsilon, args.num_slices, args.slice_tiles, tensor_args.sum.logical_shape());
}

}  // namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device

namespace ttnn::prim {

ttml::metal::ops::add_rmsnorm_fw_allcore::device::AddRMSNormForwardPartialDeviceOperation::tensor_return_value_t
ttml_add_rmsnorm_fw_allcore_partial(
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& delta_tensor,
    const ttml::metal::ops::rmsnorm_bw_allcore::device::WorkSplit& split) {
    using OperationType = ttml::metal::ops::add_rmsnorm_fw_allcore::device::AddRMSNormForwardPartialDeviceOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{.num_slices = split.num_slices, .slice_tiles = split.slice_tiles};
    auto tensor_args = OperationType::tensor_args_t{.residual = residual_tensor, .delta = delta_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttml::metal::ops::add_rmsnorm_fw_allcore::device::AddRMSNormForwardDeviceOperation::tensor_return_value_t
ttml_add_rmsnorm_fw_allcore(
    const ttnn::Tensor& sum_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& partials_tensor,
    const ttml::metal::ops::rmsnorm_bw_allcore::device::WorkSplit& split,
    float epsilon) {
    using OperationType = ttml::metal::ops::add_rmsnorm_fw_allcore::device::AddRMSNormForwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon, .num_slices = split.num_slices, .slice_tiles = split.slice_tiles};
    auto tensor_args =
        OperationType::tensor_args_t{.sum = sum_tensor, .gamma = gamma_tensor, .partials = partials_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
