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
        {DataType::BFLOAT8_B, &packed_buffer_size_bytes<bfloat8_b>}
    };
    return packed_buffer_size_bytes_map.at(dtype)(volume_unpacked_data);
}

void initialize_data_wrapper(Tensor &tensor, Initialize init_type) {
    const static std::map<DataType, std::function<void(Tensor &, Initialize)>> initialize_data_map = {
        {DataType::BFLOAT16, &initialize_data_helper<bfloat16>},
        {DataType::FLOAT32, &initialize_data_helper<float>},
        {DataType::UINT32, &initialize_data_helper<uint32_t>},
        {DataType::BFLOAT8_B, &initialize_data_helper<float>}
    };
    return initialize_data_map.at(tensor.dtype())(tensor, init_type);
}

Tensor to_host_wrapper(const Tensor &tensor) {
    const static std::map<DataType, std::function<Tensor(const Tensor &)>> to_host_map = {
        {DataType::BFLOAT16, &to_host<bfloat16>},
        {DataType::FLOAT32, &to_host<float>},
        {DataType::UINT32, &to_host<uint32_t>},
        {DataType::BFLOAT8_B, &to_host<float>}
    };
    return to_host_map.at(tensor.dtype())(tensor);
}

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device, const MemoryConfig &mem_config) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, Device *, const MemoryConfig &)>> to_device_map = {
        {DataType::BFLOAT16, &to_device<bfloat16>},
        {DataType::FLOAT32, &to_device<float>},
        {DataType::UINT32, &to_device<uint32_t>},
        {DataType::BFLOAT8_B, &to_device<float>}
    };
    return to_device_map.at(tensor.dtype())(tensor, target_device, mem_config);
}

Tensor to_layout_wrapper(const Tensor &tensor, Layout target_layout) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, Layout)>> to_layout_map = {
        {DataType::BFLOAT16, &to_layout<bfloat16>},
        {DataType::FLOAT32, &to_layout<float>},
        {DataType::UINT32, &to_layout<uint32_t>},
        {DataType::BFLOAT8_B, &to_layout<float>}
    };
    return to_layout_map.at(tensor.dtype())(tensor, target_layout);
}

Tensor pad_wrapper(const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, const std::array<uint32_t, 4> &, const std::array<uint32_t, 4> &, float)>> pad_map = {
        {DataType::BFLOAT16, &pad<bfloat16>},
        {DataType::FLOAT32, &pad<float>},
        {DataType::UINT32, &pad<uint32_t>},
        {DataType::BFLOAT8_B, &pad<float>}
    };
    return pad_map.at(tensor.dtype())(tensor, output_tensor_shape, input_tensor_start, pad_value);
}

Tensor unpad_wrapper(const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, const std::array<uint32_t, 4> &, const std::array<uint32_t, 4> &)>> unpad_map = {
        {DataType::BFLOAT16, &unpad<bfloat16>},
        {DataType::FLOAT32, &unpad<float>},
        {DataType::UINT32, &unpad<uint32_t>},
        {DataType::BFLOAT8_B, &unpad<float>}
    };
    return unpad_map.at(tensor.dtype())(tensor, output_tensor_start, output_tensor_end);
}

void print_wrapper(const Tensor &tensor, Layout print_layout, bool pretty_print) {
    const static std::map<DataType, std::function<void(const Tensor &, Layout, bool)>> print_map = {
        {DataType::BFLOAT16, &print<bfloat16>},
        {DataType::FLOAT32, &print<float>},
        {DataType::UINT32, &print<uint32_t>},
        {DataType::BFLOAT8_B, &print<float>}
    };
    print_map.at(tensor.dtype())(tensor, print_layout, pretty_print);
}

void deepcopy_host_data_wrapper(const Tensor &src, Tensor &dst) {
    const static std::map<DataType, std::function<void(const Tensor &, Tensor &)>> deepcopy_data_map = {
        {DataType::BFLOAT16, &deepcopy_host_data<bfloat16>},
        {DataType::FLOAT32, &deepcopy_host_data<float>},
        {DataType::UINT32, &deepcopy_host_data<uint32_t>},
        {DataType::BFLOAT8_B, &deepcopy_host_data<float>}
    };
    deepcopy_data_map.at(src.dtype())(src, dst);
}

void deepcopy_device_data_wrapper(const Tensor &src, Tensor &dst) {
    const static std::map<DataType, std::function<void(const Tensor &, Tensor &)>> deepcopy_dev_data_map = {
        {DataType::BFLOAT16, &deepcopy_device_data<bfloat16>},
        {DataType::FLOAT32, &deepcopy_device_data<float>},
        {DataType::UINT32, &deepcopy_device_data<uint32_t>},
        {DataType::BFLOAT8_B, &deepcopy_device_data<float>}
    };
    deepcopy_dev_data_map.at(src.dtype())(src, dst);
}

void move_host_data_wrapper(Tensor &&src, Tensor &dst) {
    const static std::map<DataType, std::function<void(Tensor &&, Tensor &)>> move_data_map = {
        {DataType::BFLOAT16, &move_host_data<bfloat16>},
        {DataType::FLOAT32, &move_host_data<float>},
        {DataType::UINT32, &move_host_data<uint32_t>},
        {DataType::BFLOAT8_B, &move_host_data<float>}
    };
    move_data_map.at(src.dtype())(std::move(src), dst);
}

void move_device_data_wrapper(Tensor &&src, Tensor &dst) {
    const static std::map<DataType, std::function<void(Tensor &&, Tensor &)>> move_dev_data_map = {
        {DataType::BFLOAT16, &move_device_data<bfloat16>},
        {DataType::FLOAT32, &move_device_data<float>},
        {DataType::UINT32, &move_device_data<uint32_t>},
        {DataType::BFLOAT8_B, &move_device_data<float>}
    };
    move_dev_data_map.at(src.dtype())(std::move(src), dst);
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
