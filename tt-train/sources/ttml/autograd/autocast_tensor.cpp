// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autocast_tensor.hpp"

#include "core/tt_tensor_utils.hpp"

namespace {

inline bool is_castable_tensor(const tt::tt_metal::Tensor &tensor) {
    return tensor.dtype() == ttnn::DataType::FLOAT32;
}

}  // namespace

namespace ttml::autograd {

void AutocastTensor::set_tensor(const tt::tt_metal::Tensor &tensor) {
    if (tensor.dtype() == ttnn::DataType::FLOAT32) {
        m_full_precision_tensor = tensor;
        m_half_precision_tensor = ttnn::typecast(tensor, ttnn::DataType::BFLOAT16);
        return;
    }

    m_full_precision_tensor = tensor;
    m_half_precision_tensor = ttnn::Tensor();  // Reset the half precision tensor
}

const tt::tt_metal::Tensor &AutocastTensor::get_tensor(PreferredPrecision preferred_precision) const {
    if (preferred_precision == PreferredPrecision::HALF && is_castable_tensor(m_full_precision_tensor)) {
        return m_half_precision_tensor;
    }

    return m_full_precision_tensor;
}

AutocastTensor::AutocastTensor(const tt::tt_metal::Tensor &tensor) {
    set_tensor(tensor);
}

}  // namespace ttml::autograd
