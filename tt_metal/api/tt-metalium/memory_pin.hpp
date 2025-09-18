// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <cstddef>

namespace tt::tt_metal {

// RAII wrapper for pinning reference-counted resources.
class MemoryPin {
public:
    MemoryPin() = default;
    MemoryPin(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count);
    MemoryPin(std::shared_ptr<void> resource);
    ~MemoryPin();

    MemoryPin(const MemoryPin& other);
    MemoryPin& operator=(const MemoryPin& other);
    MemoryPin(MemoryPin&& other) noexcept;
    MemoryPin& operator=(MemoryPin&& other) noexcept;

    friend bool operator==(const MemoryPin& pin, std::nullptr_t) noexcept;
    friend bool operator==(std::nullptr_t, const MemoryPin& pin) noexcept;
    friend bool operator!=(const MemoryPin& pin, std::nullptr_t) noexcept;
    friend bool operator!=(std::nullptr_t, const MemoryPin& pin) noexcept;

private:
    void maybe_increment();
    void maybe_decrement();

    std::function<void()> inc_;
    std::function<void()> dec_;
};

}  // namespace tt::tt_metal
