// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "autocast_tensor.hpp"

#include <tt-logger/tt-logger.hpp>

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

const tt::tt_metal::Tensor &AutocastTensor::get_tensor(PreferredPrecision preferred_precision, bool autocast) const {
    TT_FATAL(has_half() || has_full(), "AutocastTensor has neither half nor full precision tensor initialized");
    // Non-float tensors (e.g. UINT32 embedding indices) are stored in the full-precision
    // slot and returned unchanged — they cannot be typecast to half/full float precision.
    if (has_full() && !is_float_dtype(m_full_precision_tensor.dtype())) {
        return m_full_precision_tensor;
    }

    if (!autocast) {
        if (preferred_precision == PreferredPrecision::HALF) {
            if (!has_half()) {
                log_warning(
                    tt::LogAlways,
                    "Requested half precision tensor but AutocastTensor has no half precision tensor initialized. "
                    "Since autocast=false, returning full precision tensor instead. To create a half precision tensor, "
                    "set autocast=true.");
                return m_full_precision_tensor;
            }
            return m_half_precision_tensor;
        }
        if (!has_full()) {
            log_warning(
                tt::LogAlways,
                "Requested full precision tensor but AutocastTensor has no full precision tensor initialized. Since "
                "autocast=false, returning half precision tensor instead. To create a full precision tensor, set "
                "autocast=true.");
            return m_half_precision_tensor;
        }
        return m_full_precision_tensor;
    }

    // TODO: Lazy precision caching can leave the FULL/FLOAT32 view stale
    // after in-place updates that mutate only the BF16 tensor (e.g. optimizer step).
    // Revisit cache invalidation/refresh strategy so both views stay coherent.
    // Tracking: #41657

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
