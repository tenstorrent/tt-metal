// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace tt::tt_metal {

class MemoryPinImpl;

// RAII wrapper for pinning reference-counted resources.
class MemoryPin {
public:
    MemoryPin();
    explicit MemoryPin(MemoryPinImpl);
    MemoryPin(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count);
    MemoryPin(std::shared_ptr<void> resource);

    ~MemoryPin();

    MemoryPin(const MemoryPin& other);
    MemoryPin& operator=(const MemoryPin& other);
    MemoryPin(MemoryPin&& other) noexcept;
    MemoryPin& operator=(MemoryPin&& other) noexcept;

    const MemoryPinImpl& impl() const;
    MemoryPinImpl& impl();

    friend bool operator==(const MemoryPin& pin, std::nullptr_t) noexcept;
    friend bool operator==(std::nullptr_t, const MemoryPin& pin) noexcept;
    friend bool operator!=(const MemoryPin& pin, std::nullptr_t) noexcept;
    friend bool operator!=(std::nullptr_t, const MemoryPin& pin) noexcept;

private:
    std::unique_ptr<MemoryPinImpl> impl_;
};

}  // namespace tt::tt_metal
