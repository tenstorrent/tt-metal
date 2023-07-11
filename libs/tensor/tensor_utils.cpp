#include "tensor/tensor_utils.hpp"
#include "tensor/host_buffer.hpp"

namespace tt {

namespace tt_metal {

    template <typename T>
    Tensor to_weight_tile_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w) {
        auto w_shape = conv_weight_tensor.shape();
        auto data = host_buffer::view_as<T>(conv_weight_tensor);
        auto weight_matrix_cols = w_shape[0];
        // width padding
        uint32_t in1_block_w_datums = in1_block_w * 32;
        if(weight_matrix_cols%in1_block_w_datums != 0) {
            weight_matrix_cols = (uint32_t) std::ceil( (double) weight_matrix_cols / (double) in1_block_w_datums ) * in1_block_w_datums;
        }
        // height padding
        auto weight_matrix_rows = w_shape[1]*w_shape[2]*w_shape[3];
        uint32_t in1_block_h_datums = in1_block_h * 32;
        if (weight_matrix_rows % in1_block_h_datums != 0) {
            weight_matrix_rows = (uint32_t) std::ceil( (double) weight_matrix_rows / (double) in1_block_h_datums ) * in1_block_h_datums;
        }
        std::array<uint32_t, 4> output_shape = {1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = host_buffer::create<T>(volume(output_shape));
        auto output_view = host_buffer::view_as<T>(output_buffer);
        for(auto r = 0; r < w_shape[2]; r++) {
            for(auto s = 0; s < w_shape[3]; s++) {
                for(auto c = 0; c < w_shape[1]; c++) {
                    for(auto k = 0; k < weight_matrix_cols; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols + r * w_shape[3] * w_shape[1] * weight_matrix_cols;
                        if (k < w_shape[0]) {
                            auto idx = k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
                            output_view[matrix_idx] = data[idx];
                        }
                    }
                }
            }
        }
        auto rm_tensor = Tensor(HostStorage{output_buffer}, output_shape, conv_weight_tensor.dtype(), Layout::ROW_MAJOR);
        return rm_tensor.to(Layout::TILE);
    }

    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w) {
        TT_ASSERT(conv_weight_tensor.layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for conversion to tilized layout.");
        const static std::map<DataType, std::function<Tensor(const Tensor &, uint32_t in1_block_h, uint32_t in1_block_w)>> to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout<float>},
            {DataType::UINT32, &to_weight_tile_layout<uint32_t>}
        };
        return to_w_tile_layout_map.at(conv_weight_tensor.dtype())(conv_weight_tensor, in1_block_h, in1_block_w);
    }

const std::array<uint32_t, 4> infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume) {
    vector<int> ns{N, C, H, W};
    int neg_idx = -1;
    for (int i = 0; i < ns.size(); i++) {
        if (ns[i] == -1) {
            TT_ASSERT(neg_idx == -1, "Only one -1 is allowed in reshape");
            neg_idx = i;
        } else {
            TT_ASSERT(ns[i] > 0, "New shape entries can only have -1 or positive values");
        }
    }

    switch (neg_idx) {
        case 0:
            TT_ASSERT(old_volume % C*H*W == 0);
            N = old_volume/(C*H*W);
            break;
        case 1:
            TT_ASSERT(old_volume % N*H*W == 0);
            C = old_volume/(N*H*W);
            break;
        case 2:
            TT_ASSERT(old_volume % N*C*W == 0);
            H = old_volume/(N*C*W);
            TT_ASSERT(H%32 == 0);
            break;
        case 3:
            TT_ASSERT(old_volume % N*C*H == 0);
            W = old_volume/(N*C*H);
            TT_ASSERT(W%32 == 0);
            break;
        case -1: // In case where there is no negative value in ns
            TT_ASSERT(N*C*H*W == old_volume);
            break;
        default:
            TT_ASSERT(false && "Unexpected neg_idx in reshape!");
    }

    return {(uint32_t)N, (uint32_t)C, (uint32_t)H, (uint32_t)W};
}

}

}
