// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_impl_wrapper.hpp"

#include "common/bfloat16.hpp"
#include <bitset>

namespace tt {

namespace tt_metal {

namespace tensor_impl {

uint32_t element_size_bytes_wrapper(DataType dtype) {
    const static std::map<DataType, std::function<uint32_t()>> element_size_bytes_map = {
        {DataType::BFLOAT16, &element_size_bytes<bfloat16>},
        {DataType::FLOAT32, &element_size_bytes<float>},
        {DataType::UINT32, &element_size_bytes<uint32_t>},
        {DataType::UINT16, &element_size_bytes<uint16_t>},
        {DataType::BFLOAT8_B, &element_size_bytes<std::byte>},
        {DataType::BFLOAT4_B, &element_size_bytes<std::byte>},
    };
    return element_size_bytes_map.at(dtype)();
}

uint32_t packed_buffer_size_bytes_wrapper(DataType dtype, uint32_t volume_unpacked_data) {
    const static std::map<DataType, std::function<uint32_t(uint32_t)>> packed_buffer_size_bytes_map = {
        {DataType::BFLOAT16, &packed_buffer_size_bytes<bfloat16>},
        {DataType::FLOAT32, &packed_buffer_size_bytes<float>},
        {DataType::UINT32, &packed_buffer_size_bytes<uint32_t>},
        {DataType::BFLOAT8_B, &packed_buffer_size_bytes<uint32_t>},
        {DataType::BFLOAT4_B, &packed_buffer_size_bytes<uint32_t>},
        {DataType::UINT16, &packed_buffer_size_bytes<uint16_t>},
    };
    return packed_buffer_size_bytes_map.at(dtype)(volume_unpacked_data);
}

Tensor to_host_wrapper(const Tensor &tensor, bool blocking) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, bool blocking)>> to_host_map = {
        {DataType::BFLOAT16, &to_host<bfloat16>},
        {DataType::FLOAT32, &to_host<float>},
        {DataType::UINT32, &to_host<uint32_t>},
        {DataType::BFLOAT8_B, &to_host<uint32_t>},
        {DataType::BFLOAT4_B, &to_host<uint32_t>},
        {DataType::UINT16, &to_host<uint16_t>},
    };
    return to_host_map.at(tensor.get_dtype())(tensor, blocking);
}


Tensor to_extract_shard_wrapper(const Tensor &tensor, const uint32_t & core_id) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, const uint32_t &)>> to_host_map = {
        {DataType::BFLOAT16, &extract_shard<bfloat16>},
        {DataType::FLOAT32, &extract_shard<float>},
        {DataType::UINT32, &extract_shard<uint32_t>},
        {DataType::BFLOAT8_B, &extract_shard<uint32_t>},
        {DataType::BFLOAT4_B, &extract_shard<uint32_t>},
        {DataType::UINT16, &extract_shard<uint16_t>},
    };
    return to_host_map.at(tensor.get_dtype())(tensor, core_id);
}

Tensor to_host_wrapper_sharded(const Tensor &tensor) {
    const static std::map<DataType, std::function<Tensor(const Tensor &)>> to_host_map = {
        {DataType::BFLOAT16, &to_host_sharded<bfloat16>},
        {DataType::FLOAT32, &to_host_sharded<float>},
        {DataType::UINT32, &to_host_sharded<uint32_t>},
        {DataType::BFLOAT8_B, &to_host_sharded<uint32_t>},
        {DataType::BFLOAT4_B, &to_host_sharded<uint32_t>},
        {DataType::UINT16, &to_host_sharded<uint16_t>},
    };
    return to_host_map.at(tensor.get_dtype())(tensor);
}

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device, const MemoryConfig &mem_config, std::optional< std::reference_wrapper<CommandQueue> > q) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor &, Device *, const MemoryConfig &, std::optional<std::reference_wrapper<CommandQueue>> )>>
        to_device_map = {
            {DataType::BFLOAT16, &to_device<bfloat16>},
            {DataType::FLOAT32, &to_device<float>},
            {DataType::UINT32, &to_device<uint32_t>},
            {DataType::BFLOAT8_B, &to_device<uint32_t>},
            {DataType::BFLOAT4_B, &to_device<uint32_t>},
            {DataType::UINT16, &to_device<uint16_t>},
        };
    return to_device_map.at(tensor.get_dtype())(tensor, target_device, mem_config, q);
}


Tensor to_layout_wrapper(const Tensor &tensor, Layout target_layout) {
    if (tensor.get_dtype() == DataType::BFLOAT8_B || tensor.get_dtype() == DataType::BFLOAT4_B) {
        if (target_layout == Layout::TILE) {
            return tensor;
        } else {
            TT_THROW("BFLOAT8_B and BFLOAT4_B tensors can only be in TILE layout");
        }
    }

    const static std::unordered_map<DataType, std::function<Tensor(const Tensor &, Layout)>> to_layout_map = {
        {DataType::BFLOAT16, &to_layout<bfloat16>},
        {DataType::FLOAT32, &to_layout<float>},
        {DataType::UINT32, &to_layout<uint32_t>},
        {DataType::UINT16, &to_layout<uint16_t>},
    };
    return to_layout_map.at(tensor.get_dtype())(tensor, target_layout);
}

Tensor pad_wrapper(const Tensor &tensor, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) {
    if (tensor.get_dtype() == DataType::BFLOAT8_B || tensor.get_dtype() == DataType::BFLOAT4_B) {
        TT_THROW("BFLOAT8_B and BFLOAT4_B tensors cannot be padded on the host");
    }

    const static std::
        unordered_map<DataType, std::function<Tensor(const Tensor &, const Shape &, const Shape &, float)>>
            pad_map = {
                {DataType::BFLOAT16, &pad<bfloat16>},
                {DataType::FLOAT32, &pad<float>},
                {DataType::UINT32, &pad<uint32_t>},
                {DataType::UINT16, &pad<uint16_t>},
            };
    return pad_map.at(tensor.get_dtype())(tensor, output_tensor_shape, input_tensor_start, pad_value);
}

Tensor unpad_wrapper(const Tensor &tensor, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    if (tensor.get_dtype() == DataType::BFLOAT8_B || tensor.get_dtype() == DataType::BFLOAT4_B) {
        TT_THROW("BFLOAT8_B and BFLOAT4_B tensors cannot be unpadded on the host");
    }

    const static std::unordered_map<DataType, std::function<Tensor(const Tensor &, const Shape &, const Shape &)>>
        unpad_map = {
            {DataType::BFLOAT16, &unpad<bfloat16>},
            {DataType::FLOAT32, &unpad<float>},
            {DataType::UINT32, &unpad<uint32_t>},
            {DataType::UINT16, &unpad<uint16_t>},
        };
    return unpad_map.at(tensor.get_dtype())(tensor, output_tensor_start, output_tensor_end);
}

std::string to_string_wrapper(const Tensor &tensor) {
    const static std::unordered_map<DataType, std::function<std::string(const Tensor &, std::optional<DataType>)>>
        to_string_map = {
            {DataType::BFLOAT16, &to_string<bfloat16>},
            {DataType::FLOAT32, &to_string<float>},
            {DataType::UINT32, &to_string<uint32_t>},
            {DataType::BFLOAT8_B, &to_string<uint32_t>},
            {DataType::BFLOAT4_B, &to_string<uint32_t>},
            {DataType::UINT16, &to_string<uint16_t>},
        };
    return to_string_map.at(tensor.get_dtype())(tensor, std::nullopt);
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
