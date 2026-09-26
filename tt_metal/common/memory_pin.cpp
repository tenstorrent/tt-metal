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
#include <utility>

namespace tt::tt_metal {

MemoryPinImpl::MemoryPinImpl(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count) :
    inc_(std::move(increment_ref_count)),
    dec_(std::move(decrement_ref_count)),
    final_release_state_(std::make_shared<FinalReleaseState>()) {
    maybe_increment();
}

MemoryPinImpl::MemoryPinImpl(std::shared_ptr<void> resource) :
    inc_([]() {}),
    dec_([ref = std::move(resource)]() mutable { ref.reset(); }),
    final_release_state_(std::make_shared<FinalReleaseState>()) {}

void MemoryPinImpl::add_final_release_callback(std::function<void()> callback) {
    if (!final_release_state_) {
        final_release_state_ = std::make_shared<FinalReleaseState>();
    }
    final_release_state_->callbacks.push_back(std::move(callback));
}

void MemoryPinImpl::maybe_increment() {
    if (inc_) {
        inc_();
    }
}

void MemoryPinImpl::maybe_decrement() {
    release_final_release_state();
    if (dec_) {
        dec_();
    }
}

MemoryPinImpl::MemoryPinImpl(const MemoryPinImpl& other) :
    inc_(other.inc_), dec_(other.dec_), final_release_state_(other.final_release_state_) {
    acquire_final_release_state();
}

MemoryPinImpl& MemoryPinImpl::operator=(const MemoryPinImpl& other) {
    if (this != &other) {
        inc_ = other.inc_;
        dec_ = other.dec_;
        final_release_state_ = other.final_release_state_;
        acquire_final_release_state();
    }
    return *this;
}

void MemoryPinImpl::acquire_final_release_state() {
    if (final_release_state_) {
        final_release_state_->holders.fetch_add(1, std::memory_order_relaxed);
    }
}

void MemoryPinImpl::release_final_release_state() {
    if (!final_release_state_) {
        return;
    }
    // acq_rel: callbacks added through any copy happen-before the release that runs them.
    if (final_release_state_->holders.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        for (const auto& callback : final_release_state_->callbacks) {
            callback();
        }
    }
    final_release_state_.reset();
}

void MemoryPinImpl::mark_device_immutable() {
    TT_FATAL(!is_empty(), "Cannot mark an empty MemoryPin device-immutable: it keeps no memory alive.");
    if (!final_release_state_) {
        final_release_state_ = std::make_shared<FinalReleaseState>();
    }
    final_release_state_->device_immutable = true;
}

bool MemoryPinImpl::is_device_immutable() const noexcept {
    return final_release_state_ != nullptr && final_release_state_->device_immutable;
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
