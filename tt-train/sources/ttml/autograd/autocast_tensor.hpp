// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <cstdint>

namespace ttml::autograd {

enum class PreferredPrecision : uint8_t { HALF = 0, FULL = 1 };

class AutocastTensor {
    tt::tt_metal::Tensor m_half_precision_tensor{};
    tt::tt_metal::Tensor m_full_precision_tensor{};

public:
    AutocastTensor() = default;
    explicit AutocastTensor(const tt::tt_metal::Tensor &tensor);
    AutocastTensor(const AutocastTensor &) = default;
    AutocastTensor(AutocastTensor &&) noexcept = default;
    AutocastTensor &operator=(const AutocastTensor &) = default;
    AutocastTensor &operator=(AutocastTensor &&) noexcept = default;
    ~AutocastTensor() = default;

    void set_tensor(const tt::tt_metal::Tensor &tensor);
    [[nodiscard]] const tt::tt_metal::Tensor &get_tensor(
        PreferredPrecision preferred_precision = PreferredPrecision::HALF) const;
};

}  // namespace ttml::autograd
