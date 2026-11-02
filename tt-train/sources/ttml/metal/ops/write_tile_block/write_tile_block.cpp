// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "write_tile_block.hpp"

#include "device/write_tile_block_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor write_tile_block(
    const ttnn::Tensor& destination, const ttnn::Tensor& source, uint32_t tile_row_offset, uint32_t tile_col_offset) {
    return ttnn::prim::ttml_write_tile_block(destination, source, tile_row_offset, tile_col_offset);
}

}  // namespace ttml::metal
