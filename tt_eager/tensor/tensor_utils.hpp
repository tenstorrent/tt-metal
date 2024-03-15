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
        auto volume = 1;
        for (auto index = 0; index < shape.rank(); index++) {
            volume *= shape[index];
        }
        return volume;
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
        if (data_type == DataType::BFLOAT4_B) {
            TT_ASSERT(volume % constants::TILE_HW == 0);
            const auto bfloat4_b_volume = volume / constants::TILE_HW * constants::BFLOAT4_B_TILE_HW;
            TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
            return bfloat4_b_volume / sizeof(std::uint32_t);
        }
        return volume;
    }

   bool is_arch_gs(const tt::ARCH& arch);
   bool is_arch_whb0(const tt::ARCH& arch);

   bool is_cpu_tensor(const Tensor& tensor);
   bool is_device_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns the tensor on the given device.
Tensor get_device_tensor(const Device* device, const Tensor& multi_device_tensor);

// Given a multi-device tensor, return all the devices it is mapped to.
std::vector<Device*> get_devices(const Tensor& multi_device_tensor);

template<typename TensorContainer>
auto get_device_tensors(Device* device, const TensorContainer& input_tensors) {
    // Could be Tensor, const Tensor, std::optional<Tensor>, or std::optional<const Tensor>
    using ValueType = typename TensorContainer::value_type;

    // We need a way to extract the underlying Tensor type (const or non-const) from ValueType
    // and to decide whether we are dealing with an optional type.
    using IsOptional = std::conditional_t<std::is_same_v<ValueType, std::optional<Tensor>> || std::is_same_v<ValueType, std::optional<const Tensor>>,
                                        std::true_type, std::false_type>;
    using TensorType = std::conditional_t<std::is_same_v<ValueType, std::optional<Tensor>> || std::is_same_v<ValueType, Tensor>,
                                        Tensor, const Tensor>;

    // Result container type adjustment based on input type
    using ResultType = std::conditional_t<IsOptional::value, std::optional<TensorType>, TensorType>;
    std::vector<ResultType> transformed_tensors;

    for (const auto& tensor : input_tensors) {
        if constexpr (IsOptional::value) {
            if (tensor.has_value()) {
                transformed_tensors.push_back(get_device_tensor(device, *tensor));
            } else {
                transformed_tensors.push_back(std::nullopt);
            }
        } else {
            transformed_tensors.push_back(get_device_tensor(device, tensor));
        }
    }
    return transformed_tensors;
}
} // namespace tt_metal

} // namespace tt
