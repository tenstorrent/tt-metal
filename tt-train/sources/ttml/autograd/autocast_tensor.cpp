// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autocast_tensor.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

void AutocastTensor::set_tensor(const tt::tt_metal::Tensor &tensor) {
    if (tensor.dtype() == ttnn::DataType::FLOAT32) {
        m_full_precision_tensor = tensor;
        m_half_precision_tensor = ttnn::Tensor();
    } else {
        m_half_precision_tensor = tensor;
        m_full_precision_tensor = ttnn::Tensor();
    }
}

const tt::tt_metal::Tensor &AutocastTensor::get_tensor(PreferredPrecision preferred_precision) const {
    if (preferred_precision == PreferredPrecision::HALF) {
        if (!core::is_tensor_initialized(m_half_precision_tensor)) {
            m_half_precision_tensor = ttnn::typecast(m_full_precision_tensor, ttnn::DataType::BFLOAT16);
        }
        return m_half_precision_tensor;
    }

    if (!core::is_tensor_initialized(m_full_precision_tensor)) {
        m_full_precision_tensor = ttnn::typecast(m_half_precision_tensor, ttnn::DataType::FLOAT32);
    }
    return m_full_precision_tensor;
}

AutocastTensor::AutocastTensor(const tt::tt_metal::Tensor &tensor) {
    set_tensor(tensor);
}

}  // namespace ttml::autograd
