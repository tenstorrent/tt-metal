// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/memory_pin.hpp>

#include <functional>
#include <utility>
#include <cstddef>

namespace tt::tt_metal {

MemoryPin::MemoryPin(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count) :
    inc_(std::move(increment_ref_count)), dec_(std::move(decrement_ref_count)) {
    maybe_increment();
}

MemoryPin::MemoryPin(std::shared_ptr<void> resource) :
    inc_([]() {}), dec_([ref = std::move(resource)]() mutable { ref.reset(); }) {}

MemoryPin::~MemoryPin() { maybe_decrement(); }

MemoryPin::MemoryPin(const MemoryPin& other) : inc_(other.inc_), dec_(other.dec_) { maybe_increment(); }

MemoryPin& MemoryPin::operator=(const MemoryPin& other) {
    if (this != &other) {
        maybe_decrement();
        inc_ = other.inc_;
        dec_ = other.dec_;
        maybe_increment();
    }
    return *this;
}

MemoryPin::MemoryPin(MemoryPin&& other) noexcept : inc_(std::move(other.inc_)), dec_(std::move(other.dec_)) {
    other.inc_ = nullptr;
    other.dec_ = nullptr;
}

MemoryPin& MemoryPin::operator=(MemoryPin&& other) noexcept {
    if (this != &other) {
        maybe_decrement();
        inc_ = std::move(other.inc_);
        dec_ = std::move(other.dec_);
        other.inc_ = nullptr;
        other.dec_ = nullptr;
    }
    return *this;
}

void MemoryPin::maybe_increment() {
    if (inc_) {
        inc_();
    }
}

void MemoryPin::maybe_decrement() {
    if (dec_) {
        dec_();
    }
}

bool operator==(const MemoryPin& pin, std::nullptr_t) noexcept { return !pin.inc_ && !pin.dec_; }
bool operator==(std::nullptr_t, const MemoryPin& pin) noexcept { return pin == nullptr; }
bool operator!=(const MemoryPin& pin, std::nullptr_t) noexcept { return !(pin == nullptr); }
bool operator!=(std::nullptr_t, const MemoryPin& pin) noexcept { return !(nullptr == pin); }

}  // namespace tt::tt_metal
