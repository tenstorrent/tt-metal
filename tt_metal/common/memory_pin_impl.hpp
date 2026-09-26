// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace tt::tt_metal {

class MemoryPinImpl {
public:
    MemoryPinImpl() = default;
    MemoryPinImpl(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count);
    explicit MemoryPinImpl(std::shared_ptr<void> resource);

    // A copy shares the final-release state and counts as one more holder of it; a move transfers the holder.
    MemoryPinImpl(const MemoryPinImpl& other);
    MemoryPinImpl& operator=(const MemoryPinImpl& other);
    MemoryPinImpl(MemoryPinImpl&& other) noexcept = default;
    MemoryPinImpl& operator=(MemoryPinImpl&& other) noexcept = default;
    ~MemoryPinImpl() = default;

    void add_final_release_callback(std::function<void()> callback);

    void maybe_increment();
    void maybe_decrement();

    bool is_empty() const noexcept;

    // See experimental::MemoryPinMarkDeviceImmutable. Shared by every copy.
    void mark_device_immutable();
    bool is_device_immutable() const noexcept;

private:
    struct FinalReleaseState {
        std::vector<std::function<void()>> callbacks;
        // Copies sharing this state. Copies may be released on different threads; the one that drops the count to
        // zero runs the callbacks, exactly once.
        std::atomic<size_t> holders{1};
        bool device_immutable = false;
    };

    void acquire_final_release_state();
    void release_final_release_state();

    std::function<void()> inc_;
    std::function<void()> dec_;
    std::shared_ptr<FinalReleaseState> final_release_state_;
};

}  // namespace tt::tt_metal
