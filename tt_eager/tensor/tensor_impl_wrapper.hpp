// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

uint32_t element_size_bytes_wrapper(DataType dtype);

uint32_t packed_buffer_size_bytes_wrapper(DataType dtype, uint32_t volume_unpacked_data);

Tensor to_host_wrapper(const Tensor &tensor);

Tensor to_host_wrapper_sharded(const Tensor &tensor);

Tensor to_extract_shard_wrapper(const Tensor &tensor, const uint32_t & core_id);

Tensor to_device_wrapper(const Tensor &tensor, Device *target_device, const MemoryConfig &mem_config);

Tensor to_layout_wrapper(const Tensor &tensor, Layout target_layout);

Tensor pad_wrapper(const Tensor &tensor, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value);

Tensor unpad_wrapper(const Tensor &tensor, const Shape &output_tensor_start, const Shape &output_tensor_end);

std::string to_string_wrapper(const Tensor &tensor, Layout print_layout, bool pretty_print);

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
