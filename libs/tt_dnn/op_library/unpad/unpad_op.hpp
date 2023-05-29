#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

Tensor unpad (const Tensor &a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end);

bool check_unpad_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end);

}  // namespace tt_metal

}  // namespace tt
