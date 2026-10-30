// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_matmul.hpp"

#include "device/skinny_matmul_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor skinny_matmul(const ttnn::Tensor& x, const ttnn::Tensor& w, bool transpose_b) {
    using namespace ops::skinny_matmul::device;
    const auto& padded = x.padded_shape();
    const uint32_t rows = padded[0] * padded[1] * (padded[2] / tt::constants::TILE_HEIGHT);
    const uint32_t Kt = padded[3] / tt::constants::TILE_WIDTH;
    const auto grid = x.device()->compute_with_storage_grid_size();
    const auto split = choose_skinny_split(rows, Kt, grid.x * grid.y);
    auto partials = ttnn::prim::ttml_skinny_matmul_partial(x, w, split, transpose_b);
    return ttnn::prim::ttml_skinny_matmul_reduce(x, w, partials, split, transpose_b);
}

}  // namespace ttml::metal
