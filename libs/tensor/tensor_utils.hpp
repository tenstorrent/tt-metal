#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor);
}

}
