// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::write_tile_block::device {

struct WriteTileBlockParams {
    uint32_t tile_row_offset = 0U;
    uint32_t tile_col_offset = 0U;  // iteration 36: the block may be narrower than the destination
};

struct WriteTileBlockInputs {
    const ttnn::Tensor& destination;
    const ttnn::Tensor& source;
};

using operation_attributes_t = WriteTileBlockParams;
using tensor_args_t = WriteTileBlockInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::write_tile_block::device
