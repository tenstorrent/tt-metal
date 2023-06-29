#pragma once

#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

uint32_t element_size_bytes_wrapper(DataType dtype);

uint32_t packed_buffer_size_bytes_wrapper(DataType dtype, uint32_t volume_unpacked_data);

template <typename T>
void inline convert_and_write_data_wrapper(Tensor &tensor, std::vector<T> &data) {
    const static std::map<DataType, std::function<void(Tensor &, std::vector<T> &)>> write_data_map = {
        {DataType::BFLOAT16, &convert_and_write_data<bfloat16, T>},
        {DataType::FLOAT32, &convert_and_write_data<float, T>},
        {DataType::UINT32, &convert_and_write_data<uint32_t, T>},
        {DataType::BFLOAT8_B, &convert_and_write_data<float, T>}
    };
    write_data_map.at(tensor.dtype())(tensor, data);
}

void initialize_data_wrapper(Tensor &tensor, Initialize init_type);

Tensor to_host_wrapper(const Tensor &tensor);

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device, const MemoryConfig &mem_config);

Tensor to_layout_wrapper(const Tensor &tensor, Layout target_layout);

Tensor pad_wrapper(const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);

Tensor unpad_wrapper(const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end);

void print_wrapper(const Tensor &tensor, Layout print_layout, bool pretty_print);

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
