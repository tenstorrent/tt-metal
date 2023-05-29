#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor tilize (const Tensor &a);
Tensor tilize_with_zero_padding (const Tensor &a);
Tensor tilize_with_val_padding(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);
Tensor tilize_conv_activation (const Tensor &a, bool conv1x1 = false);

bool check_tilize_l1_size(const Tensor &a);
bool check_tilize_with_val_padding_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start);

}  // namespace tt_metal

}  // namespace tt
