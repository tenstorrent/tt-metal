#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor);

    const std::array<uint32_t, 4> infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume);
}

}
