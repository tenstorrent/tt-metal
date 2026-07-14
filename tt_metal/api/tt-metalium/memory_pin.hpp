// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

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

    // The callback runs once when the last MemoryPin sharing this state is released, before decrementing the pin.
    void add_final_release_callback(std::function<void()> callback);

    friend bool operator==(const MemoryPin& pin, std::nullptr_t) noexcept;
    friend bool operator==(std::nullptr_t, const MemoryPin& pin) noexcept;
    friend bool operator!=(const MemoryPin& pin, std::nullptr_t) noexcept;
    friend bool operator!=(std::nullptr_t, const MemoryPin& pin) noexcept;

private:
    struct FinalReleaseState {
        std::vector<std::function<void()>> callbacks;
        bool ran = false;
    };

    void maybe_increment();
    void maybe_decrement();
    void maybe_run_final_release_callbacks();

    std::function<void()> inc_;
    std::function<void()> dec_;
    std::shared_ptr<FinalReleaseState> final_release_state_;
};

}  // namespace tt::tt_metal
