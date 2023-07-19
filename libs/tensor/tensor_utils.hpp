#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w);

    const Shape infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume);

    template<typename T>
    static std::size_t volume(const T& shape) {
       return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<uint32_t>());
    }
}

}
