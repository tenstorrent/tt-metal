// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "write_tile_block_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::write_tile_block::device {

void WriteTileBlockOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& destination = tensor_args.destination;
    const auto& source = tensor_args.source;
    check_device_tensor(destination, "WriteTileBlock", "Destination");
    check_device_tensor(source, "WriteTileBlock", "Source");
    const auto dst_shape = destination.padded_shape();
    const auto src_shape = source.padded_shape();
    const uint32_t dst_tile_cols = dst_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t src_tile_cols = src_shape[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        args.tile_col_offset + src_tile_cols <= dst_tile_cols,
        "WriteTileBlock: source tile-columns {} at offset {} exceed destination tile-columns {}",
        src_tile_cols,
        args.tile_col_offset,
        dst_tile_cols);
    const uint32_t dst_tile_rows = dst_shape.volume() / dst_shape[-1] / tt::constants::TILE_HEIGHT;
    const uint32_t src_tile_rows = src_shape.volume() / src_shape[-1] / tt::constants::TILE_HEIGHT;
    TT_FATAL(
        args.tile_row_offset + src_tile_rows <= dst_tile_rows,
        "WriteTileBlock: source tile-rows {} at offset {} exceed destination tile-rows {}",
        src_tile_rows,
        args.tile_row_offset,
        dst_tile_rows);
}

WriteTileBlockOperation::spec_return_value_t WriteTileBlockOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.destination.tensor_spec();
}

WriteTileBlockOperation::tensor_return_value_t WriteTileBlockOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.destination;
}

ttsl::hash::hash_t WriteTileBlockOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<WriteTileBlockOperation>(
        args,
        tensor_args.destination.dtype(),
        tensor_args.destination.padded_shape(),
        tensor_args.source.padded_shape());
}

}  // namespace ttml::metal::ops::write_tile_block::device

namespace ttnn::prim {

ttml::metal::ops::write_tile_block::device::WriteTileBlockOperation::tensor_return_value_t ttml_write_tile_block(
    const ttnn::Tensor& destination, const ttnn::Tensor& source, uint32_t tile_row_offset, uint32_t tile_col_offset) {
    using OperationType = ttml::metal::ops::write_tile_block::device::WriteTileBlockOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.tile_row_offset = tile_row_offset, .tile_col_offset = tile_col_offset};
    auto tensor_args = OperationType::tensor_args_t{.destination = destination, .source = source};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
