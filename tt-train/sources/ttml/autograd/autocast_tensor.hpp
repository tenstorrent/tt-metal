// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <memory>

namespace ttml::autograd {

// HALF/FULL coerce to bf16/float32 (autocast compute precision). NATIVE returns the value as stored,
// with no typecast and no second copy cached.
enum class PreferredPrecision : uint8_t { HALF = 0, FULL = 1, NATIVE = 2 };

namespace detail {
struct AutocastState;
}  // namespace detail

// Write access to the native tensor of an AutocastTensor, obtained from get_value_for_update().
// Pass tensor() to the in-place kernel. Destroying the view marks the native tensor as written, so the next
// read of the other precision refreshes the derived copy.
class MutableTensorView {
public:
    MutableTensorView(const MutableTensorView &) = delete;
    MutableTensorView &operator=(const MutableTensorView &) = delete;
    MutableTensorView(MutableTensorView &&other) noexcept;
    MutableTensorView &operator=(MutableTensorView &&) = delete;
    ~MutableTensorView();

    [[nodiscard]] const ttnn::Tensor &tensor() const;

private:
    friend class AutocastTensor;
    explicit MutableTensorView(std::shared_ptr<detail::AutocastState> state);

    std::shared_ptr<detail::AutocastState> m_state;
};

// A tensor stored in its native precision (bf16 or fp32), plus a derived copy in the other float precision that
// is created on first use.
//
// - The native tensor is the source of truth. In-place writes go only to it, through get_value_for_update().
// - get_tensor() for the other precision returns the derived copy. When the native tensor has been written since
//   the copy was cast, the copy is refreshed in place first, into the same buffer. NATIVE returns the native
//   tensor as stored.
// - Copies of an AutocastTensor share storage and versioning: a write through one copy is seen by all of them.
//   set_tensor() gives this copy a new tensor and leaves the other copies unchanged. Use a deep copy for a
//   snapshot.
// - A write that bypasses get_value_for_update(), e.g. a kernel writing through a get_tensor() handle, is not seen.
// - Not thread-safe: a tensor is driven from one host thread.
//
// Non-float tensors (e.g. uint32 ids) are returned as stored for every precision.
class AutocastTensor {
public:
    AutocastTensor();
    explicit AutocastTensor(const ttnn::Tensor &tensor);
    AutocastTensor(const AutocastTensor &) = default;
    AutocastTensor(AutocastTensor &&) noexcept = default;
    AutocastTensor &operator=(const AutocastTensor &) = default;
    AutocastTensor &operator=(AutocastTensor &&) noexcept = default;
    ~AutocastTensor() = default;

    void set_tensor(const ttnn::Tensor &tensor);
    [[nodiscard]] const ttnn::Tensor &get_tensor(
        PreferredPrecision preferred_precision = PreferredPrecision::HALF) const;

    // precision must be NATIVE or the native precision itself; anything else is a TT_FATAL.
    [[nodiscard]] MutableTensorView get_value_for_update(PreferredPrecision precision = PreferredPrecision::NATIVE);

    [[nodiscard]] bool has_half() const;
    [[nodiscard]] bool has_full() const;

private:
    // Version of the native tensor. The one place to swap in a buffer-level version if ttnn ever offers one.
    [[nodiscard]] uint64_t native_version() const;

    std::shared_ptr<detail::AutocastState> m_state;
};

}  // namespace ttml::autograd
