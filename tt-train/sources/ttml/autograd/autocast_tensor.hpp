// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <cstdint>

namespace ttml::autograd {

// HALF/FULL coerce to bf16/float32 (autocast compute precision). NATIVE returns the value as stored,
// with no typecast and no second copy cached.
enum class PreferredPrecision : uint8_t { HALF = 0, FULL = 1, NATIVE = 2 };

class AutocastTensor {
    mutable ttnn::Tensor m_half_precision_tensor{};
    mutable ttnn::Tensor m_full_precision_tensor{};
    PreferredPrecision m_native_precision{PreferredPrecision::FULL};

public:
    AutocastTensor() = default;
    explicit AutocastTensor(const ttnn::Tensor &tensor);
    AutocastTensor(const AutocastTensor &) = default;
    AutocastTensor(AutocastTensor &&) noexcept = default;
    AutocastTensor &operator=(const AutocastTensor &) = default;
    AutocastTensor &operator=(AutocastTensor &&) noexcept = default;
    ~AutocastTensor() = default;

    void set_tensor(const ttnn::Tensor &tensor);
    [[nodiscard]] const ttnn::Tensor &get_tensor(
        PreferredPrecision preferred_precision = PreferredPrecision::HALF) const;

    [[nodiscard]] bool has_half() const;
    [[nodiscard]] bool has_full() const;
};

}  // namespace ttml::autograd
