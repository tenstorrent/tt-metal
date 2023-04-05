#include "tensor/tensor_utils.hpp"

namespace tt {

namespace tt_metal {

    template <typename T>
    Tensor to_weight_tile_layout(Tensor conv_weight_tensor) {
        auto w_shape = conv_weight_tensor.shape();
        auto data = *reinterpret_cast<std::vector<T>*>(conv_weight_tensor.data_ptr());
        std::array<uint32_t, 4> new_shape = {1, 1, w_shape[1]*w_shape[2]*w_shape[3], w_shape[0]};
        std::vector<T> new_data;
        for(auto r = 0; r < w_shape[2]; r++) {
            for(auto s = 0; s < w_shape[3]; s++) {
                for(auto c = 0; c < w_shape[1]; c++) {
                    for(auto k = 0; k < w_shape[0]; k++) {
                        auto idx = k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
                        new_data.push_back(data[idx]);
                    }
                }
            }
        }
        auto rm_tensor = Tensor(new_data, new_shape, conv_weight_tensor.dtype(), Layout::ROW_MAJOR);
        return rm_tensor.to(Layout::TILE);
    }

    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor) {
        TT_ASSERT(conv_weight_tensor.layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for conversion to tilized layout.");
        const static std::map<DataType, std::function<Tensor(const Tensor &)>> to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout<float>},
            {DataType::UINT32, &to_weight_tile_layout<uint32_t>}
        };
        return to_w_tile_layout_map.at(conv_weight_tensor.dtype())(conv_weight_tensor);
    }
}

}
