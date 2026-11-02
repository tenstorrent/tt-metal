// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "write_tile_block_device_operation_types.hpp"
#include "write_tile_block_program_factory.hpp"

namespace ttml::metal::ops::write_tile_block::device {

struct WriteTileBlockOperation {
    using operation_attributes_t = ttml::metal::ops::write_tile_block::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::write_tile_block::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::write_tile_block::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::write_tile_block::device::tensor_return_value_t;
    using program_factory_t = std::variant<WriteTileBlockProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::write_tile_block::device

namespace ttnn::prim {

ttml::metal::ops::write_tile_block::device::WriteTileBlockOperation::tensor_return_value_t ttml_write_tile_block(
    const ttnn::Tensor& destination, const ttnn::Tensor& source, uint32_t tile_row_offset, uint32_t tile_col_offset);

}  // namespace ttnn::prim
