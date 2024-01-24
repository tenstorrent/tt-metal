// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, std::optional<DataType> output_dtype = std::nullopt);

    // Converts convolution weights to tilized 2d matrix layout with special block height padding
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, std::optional<DataType> output_dtype = std::nullopt);

    const Shape infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume);

    const Shape infer_dims_for_reshape_RM(int N, int C, int H, int W, uint32_t old_volume);

    template<typename T>
    static std::size_t compute_volume(const T& shape) {
        return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<uint32_t>());
    }

    template<typename T>
    static std::size_t compute_buffer_size(const T& shape, DataType data_type) {
        const auto volume = compute_volume(shape);
        if (data_type == DataType::BFLOAT8_B) {
            TT_ASSERT(volume % constants::TILE_HW == 0);
            const auto bfloat8_b_volume = volume / constants::TILE_HW * constants::BFLOAT8_B_TILE_HW;
            TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
            return bfloat8_b_volume / sizeof(std::uint32_t);
        }
        return volume;
    }

   bool is_arch_gs(const tt::ARCH& arch);
   bool is_arch_whb0(const tt::ARCH& arch);

   bool is_cpu_tensor(const Tensor& tensor);
   bool is_device_tensor(const Tensor& tensor);
}

}
