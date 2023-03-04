#include "ll_buda/tensor/tensor_impl_wrapper.hpp"

#include "common/bfloat16.hpp"

namespace tt {

namespace ll_buda {

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
        {DataType::UINT32, &packed_buffer_size_bytes<uint32_t>}
    };
    return packed_buffer_size_bytes_map.at(dtype)(volume_unpacked_data);
}

void initialize_data_wrapper(Tensor &tensor, Initialize init_type) {
    const static std::map<DataType, std::function<void(Tensor &, Initialize)>> initialize_data_map = {
        {DataType::BFLOAT16, &initialize_data_helper<bfloat16>},
        {DataType::FLOAT32, &initialize_data_helper<float>},
        {DataType::UINT32, &initialize_data_helper<uint32_t>}
    };
    return initialize_data_map.at(tensor.dtype())(tensor, init_type);
}

Tensor to_host_wrapper(const Tensor &tensor) {
    const static std::map<DataType, std::function<Tensor(const Tensor &)>> to_host_map = {
        {DataType::BFLOAT16, &to_host<bfloat16>},
        {DataType::FLOAT32, &to_host<float>},
        {DataType::UINT32, &to_host<uint32_t>}
    };
    return to_host_map.at(tensor.dtype())(tensor);
}

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device) {
    const static std::map<DataType, std::function<Tensor(const Tensor &, Device *)>> to_device_map = {
        {DataType::BFLOAT16, &to_device<bfloat16>},
        {DataType::FLOAT32, &to_device<float>},
        {DataType::UINT32, &to_device<uint32_t>}
    };
    return to_device_map.at(tensor.dtype())(tensor, target_device);
}

void print_wrapper(const Tensor &tensor, Layout print_layout, bool pretty_print) {
    const static std::map<DataType, std::function<void(const Tensor &, Layout, bool)>> print_map = {
        {DataType::BFLOAT16, &print<bfloat16>},
        {DataType::FLOAT32, &print<float>},
        {DataType::UINT32, &print<uint32_t>}
    };
    print_map.at(tensor.dtype())(tensor, print_layout, pretty_print);
}

}  // namespace tensor_impl

}  // namespace ll_buda

}  // namespace tt
