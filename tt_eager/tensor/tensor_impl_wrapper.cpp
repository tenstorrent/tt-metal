// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_impl_wrapper.hpp"

#include "common/bfloat16.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

uint32_t element_size_bytes_wrapper(DataType dtype) {
    const static std::map<DataType, std::function<uint32_t()>> element_size_bytes_map = {
        {DataType::BFLOAT16, &element_size_bytes<bfloat16>},
        {DataType::FLOAT32, &element_size_bytes<float>},
        {DataType::UINT32, &element_size_bytes<uint32_t>}
    };
    return element_size_bytes_map.at(dtype)();
}

uint32_t packed_buffer_size_bytes_wrapper(DataType dtype, uint32_t volume_unpacked_data) {
    const static std::map<DataType, std::function<uint32_t(uint32_t)>> packed_buffer_size_bytes_map = {
        {DataType::BFLOAT16, &packed_buffer_size_bytes<bfloat16>},
        {DataType::FLOAT32, &packed_buffer_size_bytes<float>},
        {DataType::UINT32, &packed_buffer_size_bytes<uint32_t>},
        {DataType::BFLOAT8_B, &packed_buffer_size_bytes<uint32_t>}
    };
    return packed_buffer_size_bytes_map.at(dtype)(volume_unpacked_data);
}

Tensor to_host_wrapper(const Tensor &tensor) {
    const static std::map<DataType, std::function<Tensor(const Tensor &)>> to_host_map = {
        {DataType::BFLOAT16, &to_host<bfloat16>},
        {DataType::FLOAT32, &to_host<float>},
        {DataType::UINT32, &to_host<uint32_t>},
        {DataType::BFLOAT8_B, &to_host<uint32_t>}
    };
    return to_host_map.at(tensor.dtype())(tensor);
}

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device, const MemoryConfig &mem_config) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, Device *, const MemoryConfig &)>> to_device_map = {
        {DataType::BFLOAT16, &to_device<bfloat16>},
        {DataType::FLOAT32, &to_device<float>},
        {DataType::UINT32, &to_device<uint32_t>},
        {DataType::BFLOAT8_B, &to_device<uint32_t>}
    };
    return to_device_map.at(tensor.dtype())(tensor, target_device, mem_config);
}

Tensor to_layout_wrapper(const Tensor &tensor, Layout target_layout) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, Layout)>> to_layout_map = {
        {DataType::BFLOAT16, &to_layout<bfloat16>},
        {DataType::FLOAT32, &to_layout<float>},
        {DataType::UINT32, &to_layout<uint32_t>},
        {DataType::BFLOAT8_B, &to_layout_bfloat8_b},
    };
    return to_layout_map.at(tensor.dtype())(tensor, target_layout);
}

Tensor pad_wrapper(const Tensor &tensor, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, const Shape &, const Shape &, float)>> pad_map = {
        {DataType::BFLOAT16, &pad<bfloat16>},
        {DataType::FLOAT32, &pad<float>},
        {DataType::UINT32, &pad<uint32_t>},
        {DataType::BFLOAT8_B, &pad_bfloat8_b},
    };
    return pad_map.at(tensor.dtype())(tensor, output_tensor_shape, input_tensor_start, pad_value);
}

Tensor unpad_wrapper(const Tensor &tensor, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, const Shape &, const Shape &)>> unpad_map = {
        {DataType::BFLOAT16, &unpad<bfloat16>},
        {DataType::FLOAT32, &unpad<float>},
        {DataType::UINT32, &unpad<uint32_t>},
        {DataType::BFLOAT8_B, &unpad_bfloat8_b},
    };
    return unpad_map.at(tensor.dtype())(tensor, output_tensor_start, output_tensor_end);
}

std::string to_string_wrapper(const Tensor &tensor, Layout print_layout, bool pretty_print) {
    const static std::map<DataType, std::function<std::string(const Tensor &, Layout, bool)>> to_string_map = {
        {DataType::BFLOAT16, &to_string<bfloat16>},
        {DataType::FLOAT32, &to_string<float>},
        {DataType::UINT32, &to_string<uint32_t>},
        {DataType::BFLOAT8_B, &to_string<uint32_t>}
    };
    return to_string_map.at(tensor.dtype())(tensor, print_layout, pretty_print);
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
