// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "autocast_tensor.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

static bool is_float_dtype(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::FLOAT32 || dtype == ttnn::DataType::BFLOAT16;
}

void AutocastTensor::set_tensor(const tt::tt_metal::Tensor &tensor) {
    if (tensor.dtype() == ttnn::DataType::FLOAT32) {
        m_full_precision_tensor = tensor;
        m_half_precision_tensor = ttnn::Tensor();
    } else if (tensor.dtype() == ttnn::DataType::BFLOAT16) {
        m_half_precision_tensor = tensor;
        m_full_precision_tensor = ttnn::Tensor();
    } else {
        // Non-castable types (e.g. UINT32 for embedding indices): store as-is in the
        // full-precision slot and return unchanged from get_tensor() regardless of
        // the requested precision — typecast is not applicable to integer dtypes.
        m_full_precision_tensor = tensor;
        m_half_precision_tensor = ttnn::Tensor();
    }
}

bool AutocastTensor::has_half() const {
    return core::is_tensor_initialized(m_half_precision_tensor);
}

bool AutocastTensor::has_full() const {
    return core::is_tensor_initialized(m_full_precision_tensor);
}

const tt::tt_metal::Tensor &AutocastTensor::get_tensor(PreferredPrecision preferred_precision) const {
    // Non-float tensors (e.g. UINT32 embedding indices) are stored in the full-precision
    // slot and returned unchanged — they cannot be typecast to half/full float precision.
    if (has_full() && !is_float_dtype(m_full_precision_tensor.dtype())) {
        return m_full_precision_tensor;
    }

    if (preferred_precision == PreferredPrecision::HALF) {
        if (!has_half()) {
            m_half_precision_tensor = ttnn::typecast(m_full_precision_tensor, ttnn::DataType::BFLOAT16);
        }
        return m_half_precision_tensor;
    }

    if (!has_full()) {
        m_full_precision_tensor = ttnn::typecast(m_half_precision_tensor, ttnn::DataType::FLOAT32);
    }
    return m_full_precision_tensor;
}

AutocastTensor::AutocastTensor(const tt::tt_metal::Tensor &tensor) {
    set_tensor(tensor);
}

}  // namespace ttml::autograd
