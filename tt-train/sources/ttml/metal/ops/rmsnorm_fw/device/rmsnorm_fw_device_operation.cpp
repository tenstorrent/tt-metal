// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw_device_operation.hpp"

#include <bit>
#include <enchantum/enchantum.hpp>

#include "rmsnorm_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

namespace {

void check_tensor(const ttnn::Tensor& tensor, const std::string& name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "RMSNormForward operation requires {} to be on Device. Input storage type: {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(
        tensor.buffer() != nullptr,
        "Operands to RMSNormForward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
        name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "RMSNormForward operation requires tensor to be in Tile layout. {} tensor layout: {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == dtype,
        "RMSNormForward operation requires {} to be of {} data type, got {}",
        name,
        enchantum::to_string(dtype),
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "RMSNormForward operation requires Interleaved memory layout. {} memory layout: `{}`",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "RMSNormForward operation requires {} to be in DRAM",
        name);
}

void check_input(const ttnn::Tensor& input) {
    check_tensor(input, "Input", tt::tt_metal::DataType::BFLOAT16);
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);
}

void check_split(uint32_t num_slices, uint32_t slice_tiles, uint32_t Wt) {
    TT_FATAL(num_slices >= 1 && slice_tiles >= 1, "rmsnorm_fw: invalid work split");
    TT_FATAL(
        (num_slices - 1U) * slice_tiles < Wt && num_slices * slice_tiles >= Wt,
        "rmsnorm_fw: work split ({} slices x {} tiles) does not cover Wt={}",
        num_slices,
        slice_tiles,
        Wt);
}

uint32_t rows_tiles(const ttnn::Tensor& input) {
    const auto& padded = input.padded_shape();
    return padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Phase A
// ---------------------------------------------------------------------------------------------------------------

void RMSNormForwardPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_input(tensor_args.input);
    const uint32_t Wt = tensor_args.input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    check_split(args.num_slices, args.slice_tiles, Wt);
}

partial::spec_return_value_t RMSNormForwardPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t rows = rows_tiles(tensor_args.input);
    return tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, rows * tt::constants::TILE_HEIGHT, args.num_slices * tt::constants::TILE_WIDTH},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.input.memory_config()));
}

partial::tensor_return_value_t RMSNormForwardPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t RMSNormForwardPartialDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RMSNormForwardPartialDeviceOperation>(
        args.num_slices, args.slice_tiles, tensor_args.input.logical_shape());
}

// ---------------------------------------------------------------------------------------------------------------
// Phase B
// ---------------------------------------------------------------------------------------------------------------

void RMSNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_input(tensor_args.input);
    check_tensor(tensor_args.gamma, "Gamma", tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.partials, "Partials", tt::tt_metal::DataType::FLOAT32);

    const auto& input_shape = tensor_args.input.logical_shape();
    const auto& gamma_shape = tensor_args.gamma.logical_shape();
    TT_FATAL(
        gamma_shape.rank() == 4 && gamma_shape[0] == 1 && gamma_shape[1] == 1 && gamma_shape[2] == 1,
        "Gamma tensor must have shape [1, 1, 1, C], got shape {}",
        gamma_shape);
    TT_FATAL(
        input_shape[3] == gamma_shape[3],
        "Gamma last dim (C) must match input last dim (C): input C={}, gamma C={}",
        input_shape[3],
        gamma_shape[3]);

    const uint32_t Wt = tensor_args.input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    check_split(args.num_slices, args.slice_tiles, Wt);

    const uint32_t rows = rows_tiles(tensor_args.input);
    const auto& pshape = tensor_args.partials.logical_shape();
    TT_FATAL(
        pshape.rank() == 4 && pshape[2] == rows * tt::constants::TILE_HEIGHT &&
            pshape[3] == args.num_slices * tt::constants::TILE_WIDTH,
        "Partials tensor must be [1, 1, rows*32, num_slices*32] = [1, 1, {}, {}], got {}",
        rows * tt::constants::TILE_HEIGHT,
        args.num_slices * tt::constants::TILE_WIDTH,
        pshape);
}

spec_return_value_t RMSNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    spec_return_value_t specs;
    specs.emplace_back(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::Layout::TILE, input.memory_config()));
    if (args.return_intermediates) {
        auto rms_shape = input.logical_shape();
        rms_shape[-1] = 1U;
        specs.emplace_back(
            rms_shape, tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::Layout::TILE, input.memory_config()));
    }
    return specs;
}

tensor_return_value_t RMSNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.input.device()));
    }
    return outputs;
}

ttsl::hash::hash_t RMSNormForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RMSNormForwardDeviceOperation>(
        args.return_intermediates,
        std::bit_cast<uint32_t>(args.epsilon),
        args.num_slices,
        args.slice_tiles,
        tensor_args.input.dtype(),
        tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::rmsnorm_fw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardPartialDeviceOperation::tensor_return_value_t
ttml_rmsnorm_fw_partial(
    const ttnn::Tensor& input_tensor, const ttml::metal::ops::rmsnorm_fw::device::WorkSplit& split) {
    using OperationType = ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardPartialDeviceOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{.num_slices = split.num_slices, .slice_tiles = split.slice_tiles};
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& partials_tensor,
    const ttml::metal::ops::rmsnorm_fw::device::WorkSplit& split,
    bool return_intermediates,
    float epsilon) {
    using OperationType = ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .return_intermediates = return_intermediates,
        .epsilon = epsilon,
        .num_slices = split.num_slices,
        .slice_tiles = split.slice_tiles};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .partials = partials_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
