// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_split_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::transpose_split::device {

void TransposeSplitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    check_device_tensor(input, "TransposeSplit", "input", {.buffer_type = tt::tt_metal::BufferType::DRAM});
    const auto& s = input.logical_shape();
    TT_FATAL(s.rank() == 4 && s[0] == 1 && s[1] == 1, "TransposeSplit: input must be [1, 1, R, T], got {}", s);
    TT_FATAL(
        args.rows % tt::constants::TILE_HEIGHT == 0 && args.rows > 0,
        "TransposeSplit: rows must be a positive multiple of 32, got {}",
        args.rows);
    TT_FATAL(
        args.tail_rows >= 1 && args.tail_rows <= tt::constants::TILE_HEIGHT,
        "TransposeSplit: tail_rows must be in 1..32, got {}",
        args.tail_rows);
    TT_FATAL(
        args.dx_cols == 0 ||
            (args.dx_cols >= args.rows && args.dx_cols % tt::constants::TILE_HEIGHT == 0 && args.dx_cols <= s[2]),
        "TransposeSplit: dx_cols {} must be a multiple of 32 in [rows {}, input rows {}]",
        args.dx_cols,
        args.rows,
        s[2]);
    TT_FATAL(
        s[2] >= args.rows + args.tail_rows,
        "TransposeSplit: input has {} rows, needs rows + tail_rows = {}",
        s[2],
        args.rows + args.tail_rows);
    TT_FATAL(s[3] % tt::constants::TILE_WIDTH == 0, "TransposeSplit: T must be a multiple of 32, got {}", s[3]);
}

spec_return_value_t TransposeSplitDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const uint32_t T = input.logical_shape()[3];
    auto spec = [&](uint32_t cols) {
        return tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 1, T, cols},
            tt::tt_metal::TensorLayout(
                input.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), input.memory_config()));
    };
    return {spec(args.dx_cols != 0 ? args.dx_cols : args.rows), spec(args.tail_rows)};
}

tensor_return_value_t TransposeSplitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto [dx_spec, tail_spec] = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input.device();
    return {ttnn::create_device_tensor(dx_spec, device), ttnn::create_device_tensor(tail_spec, device)};
}

ttsl::hash::hash_t TransposeSplitDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<TransposeSplitDeviceOperation>(
        args.rows, args.tail_rows, args.tail_scale, args.dx_cols, tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::transpose_split::device

namespace ttnn::prim {

ttml::metal::ops::transpose_split::device::tensor_return_value_t ttml_transpose_split(
    const ttnn::Tensor& input, uint32_t rows, uint32_t tail_rows, float tail_scale, uint32_t dx_cols) {
    using OperationType = ttml::metal::ops::transpose_split::device::TransposeSplitDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .rows = rows, .tail_rows = tail_rows, .tail_scale = tail_scale, .dx_cols = dx_cols};
    auto tensor_args = OperationType::tensor_args_t{.input = input};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
