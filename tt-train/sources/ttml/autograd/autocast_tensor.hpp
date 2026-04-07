// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <cstdint>

namespace ttml::autograd {

// PreferredPrecision is a view-selection hint for AutocastTensor caches.
// It does not guarantee a concrete dtype conversion.
//
// - HALF prefers a BF16 view.
// - FULL prefers the canonical/original stored tensor view.
//   FULL is NOT "force FP32":
//     * canonical BF16 -> returns BF16
//     * canonical FP32 -> returns FP32
//     * non-float dtypes (e.g. UINT32) -> returned unchanged
enum class PreferredPrecision : uint8_t { HALF = 0, FULL = 1 };

class AutocastTensor {
    mutable tt::tt_metal::Tensor m_half_precision_tensor{};
    mutable tt::tt_metal::Tensor m_full_precision_tensor{};

public:
    AutocastTensor() = default;
    explicit AutocastTensor(const tt::tt_metal::Tensor &tensor);
    AutocastTensor(const AutocastTensor &) = default;
    AutocastTensor(AutocastTensor &&) noexcept = default;
    AutocastTensor &operator=(const AutocastTensor &) = default;
    AutocastTensor &operator=(AutocastTensor &&) noexcept = default;
    ~AutocastTensor() = default;

    void set_tensor(const tt::tt_metal::Tensor &tensor);
    // Returns the tensor view requested by PreferredPrecision.
    // Use an explicit typecast when a concrete output dtype is required.
    [[nodiscard]] const tt::tt_metal::Tensor &get_tensor(
        PreferredPrecision preferred_precision = PreferredPrecision::HALF) const;

    [[nodiscard]] bool has_half() const;
    [[nodiscard]] bool has_full() const;
};

}  // namespace ttml::autograd
