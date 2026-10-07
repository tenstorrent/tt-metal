// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "rmsnorm_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

namespace {

void check_tensor(const ttnn::Tensor& tensor, const std::string& name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "RMSNormBackward operation requires {} to be on Device. Input storage type: {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(
        tensor.buffer() != nullptr,
        "Operands to RMSNormBackward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
        name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "RMSNormBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == dtype,
        "RMSNormBackward operation requires {} to be of {} data type, got {}",
        name,
        enchantum::to_string(dtype),
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "RMSNormBackward operation requires Interleaved memory layout. {} memory layout: `{}`",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "RMSNormBackward operation requires {} to be in DRAM",
        name);
}

void check_main_inputs(const ttnn::Tensor& input, const ttnn::Tensor& gamma, const ttnn::Tensor& dL_dout) {
    check_tensor(input, "Input", tt::tt_metal::DataType::BFLOAT16);
    check_tensor(gamma, "Gamma", tt::tt_metal::DataType::BFLOAT16);
    check_tensor(dL_dout, "dL_dout", tt::tt_metal::DataType::BFLOAT16);

    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);
    const auto& gamma_shape = gamma.logical_shape();
    TT_FATAL(
        gamma_shape.rank() == 4 && gamma_shape[0] == 1 && gamma_shape[1] == 1 && gamma_shape[2] == 1,
        "Gamma tensor must have shape [1, 1, 1, C], got shape {}",
        gamma_shape);
    TT_FATAL(
        input_shape[3] == gamma_shape[3],
        "Gamma last dim (C) must match input last dim (C): input C={}, gamma C={}",
        input_shape[3],
        gamma_shape[3]);
    TT_FATAL(dL_dout.logical_shape() == input_shape, "dL_dout must match the input shape");
}

uint32_t rows_tiles(const ttnn::Tensor& input) {
    const auto& padded = input.padded_shape();
    return padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Phase A
// ---------------------------------------------------------------------------------------------------------------

void RMSNormBackwardPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_main_inputs(tensor_args.input, tensor_args.gamma, tensor_args.dL_dout);
    const uint32_t Wt = tensor_args.input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(args.num_slices >= 1 && args.slice_tiles >= 1, "rmsnorm_bw: invalid work split");
    TT_FATAL(
        (args.num_slices - 1U) * args.slice_tiles < Wt && args.num_slices * args.slice_tiles >= Wt,
        "rmsnorm_bw: work split ({} slices x {} tiles) does not cover Wt={}",
        args.num_slices,
        args.slice_tiles,
        Wt);
}

partial::spec_return_value_t RMSNormBackwardPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t rows = rows_tiles(tensor_args.input);
    return tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, rows * tt::constants::TILE_HEIGHT, args.num_slices * tt::constants::TILE_WIDTH},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.input.memory_config()));
}

partial::tensor_return_value_t RMSNormBackwardPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t RMSNormBackwardPartialDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RMSNormBackwardPartialDeviceOperation>(
        args.num_slices, args.slice_tiles, tensor_args.input.logical_shape());
}

// ---------------------------------------------------------------------------------------------------------------
// Phase B
// ---------------------------------------------------------------------------------------------------------------

void RMSNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_main_inputs(tensor_args.input, tensor_args.gamma, tensor_args.dL_dout);
    check_tensor(tensor_args.rms, "RMS", tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.partials, "Partials", tt::tt_metal::DataType::FLOAT32);

    const auto& input_shape = tensor_args.input.logical_shape();
    const auto& rms_shape = tensor_args.rms.logical_shape();
    TT_FATAL(
        rms_shape.rank() == 4 && rms_shape[0] == input_shape[0] && rms_shape[1] == input_shape[1] &&
            rms_shape[2] == input_shape[2] && rms_shape[3] == 1,
        "RMS tensor must have shape [B, N, S, 1] matching the input, got {}",
        rms_shape);

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

spec_return_value_t RMSNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto spec = tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    spec_return_value_t specs{spec};
    if (args.compute_dgamma) {
        // dL_dgamma is reduced over (B, N, S) outside the kernel; the kernel emits per-element components.
        specs.push_back(spec);
    }
    return specs;
}

tensor_return_value_t RMSNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.input.device()));
    }
    return outputs;
}

ttsl::hash::hash_t RMSNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RMSNormBackwardDeviceOperation>(
        args.compute_dgamma,
        args.num_slices,
        args.slice_tiles,
        tensor_args.input.dtype(),
        tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardPartialDeviceOperation::tensor_return_value_t
ttml_rmsnorm_bw_partial(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttml::metal::ops::rmsnorm_bw::device::WorkSplit& split) {
    using OperationType = ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardPartialDeviceOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{.num_slices = split.num_slices, .slice_tiles = split.slice_tiles};
    auto tensor_args =
        OperationType::tensor_args_t{.input = input_tensor, .gamma = gamma_tensor, .dL_dout = dL_dout_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& partials_tensor,
    const ttml::metal::ops::rmsnorm_bw::device::WorkSplit& split,
    float epsilon,
    bool compute_dgamma) {
    using OperationType = ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
        .compute_dgamma = compute_dgamma,
        .num_slices = split.num_slices,
        .slice_tiles = split.slice_tiles};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .rms = rms_tensor,
        .dL_dout = dL_dout_tensor,
        .partials = partials_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
