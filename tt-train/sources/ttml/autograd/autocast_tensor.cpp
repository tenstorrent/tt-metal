// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "autocast_tensor.hpp"

#include <enchantum/enchantum.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

namespace detail {

// Shared by every copy of one AutocastTensor (see the header).
struct AutocastState {
    ttnn::Tensor native{};
    // The other float precision, cast from native on first use and refreshed in place when behind.
    ttnn::Tensor derived{};
    PreferredPrecision native_precision{PreferredPrecision::FULL};
    // Bumped when a MutableTensorView of native is destroyed. derived_version is the native_version the derived
    // copy was cast from; they are compared with != so wrap-around is harmless.
    uint64_t native_version{0};
    uint64_t derived_version{0};
    // Set while a MutableTensorView is alive.
    bool write_in_progress{false};
};

}  // namespace detail

namespace {

// A fresh state holding tensor. Non-float tensors (e.g. UINT32 embedding indices) count as FULL and are returned as
// stored for every precision: typecast does not apply to them.
detail::AutocastState make_state(const ttnn::Tensor &tensor) {
    return detail::AutocastState{
        .native = tensor,
        .native_precision =
            tensor.dtype() == ttnn::DataType::BFLOAT16 ? PreferredPrecision::HALF : PreferredPrecision::FULL};
}

}  // namespace

bool is_float_dtype(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::FLOAT32 || dtype == ttnn::DataType::BFLOAT16;
}

MutableTensorView::MutableTensorView(std::shared_ptr<detail::AutocastState> state) : m_state(std::move(state)) {
}

MutableTensorView::~MutableTensorView() {
    // A moved-from view holds nothing and must not count as a write.
    if (m_state) {
        ++m_state->native_version;
        m_state->write_in_progress = false;
    }
}

const ttnn::Tensor &MutableTensorView::tensor() const {
    return m_state->native;
}

AutocastTensor::AutocastTensor() : m_state(std::make_shared<detail::AutocastState>()) {
}

AutocastTensor::AutocastTensor(const ttnn::Tensor &tensor) :
    m_state(std::make_shared<detail::AutocastState>(make_state(tensor))) {
}

void AutocastTensor::set_tensor(const ttnn::Tensor &tensor) {
    TT_FATAL(!m_state->write_in_progress, "set_tensor called while the tensor is being written in place");
    // Built before the current state changes: tensor may be a reference into it, e.g. set_tensor(get_tensor(FULL)).
    auto state = make_state(tensor);
    // Sole owner: reset in place, so a reference returned by get_tensor(NATIVE) follows the new tensor. Shared with
    // copies: detach this copy and leave the others on the old state.
    if (m_state.use_count() == 1) {
        *m_state = std::move(state);
    } else {
        m_state = std::make_shared<detail::AutocastState>(std::move(state));
    }
}

bool AutocastTensor::has_half() const {
    const auto &state = *m_state;
    return (core::is_tensor_initialized(state.native) && state.native.dtype() == ttnn::DataType::BFLOAT16) ||
           (core::is_tensor_initialized(state.derived) && state.derived.dtype() == ttnn::DataType::BFLOAT16);
}

bool AutocastTensor::has_full() const {
    const auto &state = *m_state;
    return (core::is_tensor_initialized(state.native) && state.native.dtype() != ttnn::DataType::BFLOAT16) ||
           (core::is_tensor_initialized(state.derived) && state.derived.dtype() == ttnn::DataType::FLOAT32);
}

uint64_t AutocastTensor::native_version() const {
    return m_state->native_version;
}

const ttnn::Tensor &AutocastTensor::get_tensor(PreferredPrecision preferred_precision) const {
    auto &state = *m_state;
    if (preferred_precision == PreferredPrecision::NATIVE || preferred_precision == state.native_precision ||
        !core::is_tensor_initialized(state.native) || !is_float_dtype(state.native.dtype())) {
        return state.native;
    }

    TT_FATAL(
        !state.write_in_progress,
        "Reading the {} view while the {} tensor is being written in place would return stale values",
        enchantum::to_string(preferred_precision),
        enchantum::to_string(state.native_precision));

    const auto dtype =
        preferred_precision == PreferredPrecision::HALF ? ttnn::DataType::BFLOAT16 : ttnn::DataType::FLOAT32;
    const bool has_derived = core::is_tensor_initialized(state.derived);
    if (has_derived && state.derived_version == native_version()) {
        return state.derived;
    }
    if (has_derived && state.native.storage_type() == ttnn::StorageType::DEVICE) {
        // Refresh into the existing buffer: no allocation, and the buffer address stays the same.
        ttnn::typecast(state.native, dtype, std::nullopt, state.derived);
    } else {
        // First use, or a host tensor: ttnn has no in-place typecast for host tensors.
        state.derived = ttnn::typecast(state.native, dtype);
    }
    state.derived_version = native_version();
    return state.derived;
}

MutableTensorView AutocastTensor::get_value_for_update(PreferredPrecision precision) {
    auto &state = *m_state;
    TT_FATAL(
        precision == PreferredPrecision::NATIVE || precision == state.native_precision,
        "In-place updates must target the native precision ({}), got {}",
        enchantum::to_string(state.native_precision),
        enchantum::to_string(precision));
    TT_FATAL(!state.write_in_progress, "The tensor is already being written in place");
    state.write_in_progress = true;
    return MutableTensorView(m_state);
}

std::optional<ttnn::Tensor> optional_tensor(const std::optional<MutableTensorView> &view) {
    return view ? std::optional<ttnn::Tensor>(view->tensor()) : std::nullopt;
}

}  // namespace ttml::autograd
