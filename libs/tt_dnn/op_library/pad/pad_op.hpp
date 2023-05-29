#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

Tensor pad (const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);

bool check_pad_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start);

}  // namespace tt_metal

}  // namespace tt
