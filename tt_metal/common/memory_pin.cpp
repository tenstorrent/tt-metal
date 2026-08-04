// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/memory_pin.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace tt::tt_metal {

MemoryPin::MemoryPin(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count) :
    inc_(std::move(increment_ref_count)),
    dec_(std::move(decrement_ref_count)),
    final_release_state_(std::make_shared<FinalReleaseState>()) {
    maybe_increment();
}

MemoryPin::MemoryPin(std::shared_ptr<void> resource) :
    inc_([]() {}),
    dec_([ref = std::move(resource)]() mutable { ref.reset(); }),
    final_release_state_(std::make_shared<FinalReleaseState>()) {}

MemoryPin::~MemoryPin() { maybe_decrement(); }

MemoryPin::MemoryPin(const MemoryPin& other) :
    inc_(other.inc_), dec_(other.dec_), final_release_state_(other.final_release_state_) {
    maybe_increment();
}

MemoryPin& MemoryPin::operator=(const MemoryPin& other) {
    if (this != &other) {
        maybe_decrement();
        inc_ = other.inc_;
        dec_ = other.dec_;
        final_release_state_ = other.final_release_state_;
        maybe_increment();
    }
    return *this;
}

MemoryPin::MemoryPin(MemoryPin&& other) noexcept :
    inc_(std::move(other.inc_)),
    dec_(std::move(other.dec_)),
    final_release_state_(std::move(other.final_release_state_)) {
    other.inc_ = nullptr;
    other.dec_ = nullptr;
}

MemoryPin& MemoryPin::operator=(MemoryPin&& other) noexcept {
    if (this != &other) {
        maybe_decrement();
        inc_ = std::move(other.inc_);
        dec_ = std::move(other.dec_);
        final_release_state_ = std::move(other.final_release_state_);
        other.inc_ = nullptr;
        other.dec_ = nullptr;
    }
    return *this;
}

void MemoryPin::add_final_release_callback(std::function<void()> callback) {
    if (!final_release_state_) {
        final_release_state_ = std::make_shared<FinalReleaseState>();
    }
    final_release_state_->callbacks.push_back(std::move(callback));
}

void MemoryPin::maybe_increment() {
    if (inc_) {
        inc_();
    }
}

void MemoryPin::maybe_decrement() {
    maybe_run_final_release_callbacks();
    if (dec_) {
        dec_();
    }
}

void MemoryPin::maybe_run_final_release_callbacks() {
    if (!final_release_state_ || final_release_state_.use_count() != 1 || final_release_state_->ran) {
        return;
    }
    final_release_state_->ran = true;
    for (const auto& callback : final_release_state_->callbacks) {
        callback();
    }
}

bool operator==(const MemoryPin& pin, std::nullptr_t) noexcept { return !pin.inc_ && !pin.dec_; }
bool operator==(std::nullptr_t, const MemoryPin& pin) noexcept { return pin == nullptr; }
bool operator!=(const MemoryPin& pin, std::nullptr_t) noexcept { return !(pin == nullptr); }
bool operator!=(std::nullptr_t, const MemoryPin& pin) noexcept { return !(nullptr == pin); }

}  // namespace tt::tt_metal
