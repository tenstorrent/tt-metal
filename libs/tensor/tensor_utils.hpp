#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w);

    const std::array<uint32_t, 4> infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume);

    static std::size_t volume(const std::array<uint32_t, 4>& shape) {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }
}

}
