#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

namespace tensor_impl {

uint32_t element_size_bytes_wrapper(DataType dtype);

uint32_t packed_buffer_size_bytes_wrapper(DataType dtype, uint32_t volume_unpacked_data);

void initialize_data_wrapper(Tensor &tensor, Initialize init_type);

Tensor to_host_wrapper(const Tensor &tensor);

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device);

void print_wrapper(const Tensor &tensor, Layout print_layout, bool pretty_print);

}  // namespace tensor_impl

}  // namespace ll_buda

}  // namespace tt
