// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/memory_pin.hpp>
#include "common/memory_pin_impl.hpp"
#include <tt-metalium/experimental/memory_pin_access.hpp>

#include <tt_stl/assert.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace tt::tt_metal {

MemoryPinImpl::MemoryPinImpl(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count) :
    inc_(std::move(increment_ref_count)),
    dec_(std::move(decrement_ref_count)),
    shared_state_(std::make_shared<SharedState>()) {
    maybe_increment();
}

MemoryPinImpl::MemoryPinImpl(std::shared_ptr<void> resource) :
    inc_([]() {}),
    dec_([ref = std::move(resource)]() mutable { ref.reset(); }),
    shared_state_(std::make_shared<SharedState>()) {}

MemoryPinImpl::SharedState::~SharedState() {
    for (const auto& callback : callbacks) {
        callback();
    }
}

MemoryPinImpl::SharedState& MemoryPinImpl::shared_state() {
    if (!shared_state_) {
        shared_state_ = std::make_shared<SharedState>();
    }
    return *shared_state_;
}

void MemoryPinImpl::add_final_release_callback(std::function<void()> callback) {
    auto& state = shared_state();
    std::lock_guard lock(state.callbacks_mutex);
    state.callbacks.push_back(std::move(callback));
}

void MemoryPinImpl::maybe_increment() {
    if (inc_) {
        inc_();
    }
}

void MemoryPinImpl::maybe_decrement() {
    // Dropped before dec_ so that, when this is the last copy, the final-release callbacks run first.
    shared_state_.reset();
    if (dec_) {
        dec_();
    }
}

void MemoryPinImpl::mark_device_immutable() { shared_state().device_immutable.store(true, std::memory_order_release); }

bool MemoryPinImpl::is_device_immutable() const noexcept {
    return shared_state_ != nullptr && shared_state_->device_immutable.load(std::memory_order_acquire);
}

bool MemoryPinImpl::is_empty() const noexcept { return !inc_ && !dec_; }

MemoryPin::MemoryPin() = default;

MemoryPin::MemoryPin(MemoryPinImpl impl) : impl_(std::make_unique<MemoryPinImpl>(std::move(impl))) {}

MemoryPin::MemoryPin(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count) :
    impl_(std::make_unique<MemoryPinImpl>(std::move(increment_ref_count), std::move(decrement_ref_count))) {}

MemoryPin::MemoryPin(std::shared_ptr<void> resource) : impl_(std::make_unique<MemoryPinImpl>(std::move(resource))) {}

MemoryPin::~MemoryPin() {
    if (impl_) {
        impl_->maybe_decrement();
    }
}

MemoryPin::MemoryPin(const MemoryPin& other) {
    if (other.impl_) {
        impl_ = std::make_unique<MemoryPinImpl>(*other.impl_);
        impl_->maybe_increment();
    }
}

MemoryPin& MemoryPin::operator=(const MemoryPin& other) {
    if (this != &other) {
        if (impl_) {
            impl_->maybe_decrement();
        }
        if (other.impl_) {
            impl_ = std::make_unique<MemoryPinImpl>(*other.impl_);
            impl_->maybe_increment();
        } else {
            impl_.reset();
        }
    }
    return *this;
}

MemoryPin::MemoryPin(MemoryPin&& other) noexcept : impl_(std::move(other.impl_)) {}

MemoryPin& MemoryPin::operator=(MemoryPin&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            impl_->maybe_decrement();
        }
        impl_ = std::move(other.impl_);
    }
    return *this;
}

const MemoryPinImpl& MemoryPin::impl() const {
    TT_FATAL(impl_ != nullptr, "MemoryPin impl is null");
    return *impl_;
}

MemoryPinImpl& MemoryPin::impl() {
    TT_FATAL(impl_ != nullptr, "MemoryPin impl is null");
    return *impl_;
}

bool operator==(const MemoryPin& pin, std::nullptr_t) noexcept { return pin.impl_ == nullptr || pin.impl_->is_empty(); }
bool operator==(std::nullptr_t, const MemoryPin& pin) noexcept { return pin == nullptr; }
bool operator!=(const MemoryPin& pin, std::nullptr_t) noexcept { return !(pin == nullptr); }
bool operator!=(std::nullptr_t, const MemoryPin& pin) noexcept { return !(nullptr == pin); }

namespace experimental {

void MemoryPinMarkDeviceImmutable(MemoryPin& pin) {
    TT_FATAL(pin != nullptr, "Cannot mark an empty MemoryPin device-immutable: it keeps no memory alive.");
    pin.impl().mark_device_immutable();
}

bool MemoryPinIsDeviceImmutable(const MemoryPin& pin) { return pin != nullptr && pin.impl().is_device_immutable(); }

}  // namespace experimental

}  // namespace tt::tt_metal
