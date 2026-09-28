// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace tt::tt_metal {

class MemoryPinImpl {
public:
    MemoryPinImpl() = default;
    MemoryPinImpl(std::function<void()> increment_ref_count, std::function<void()> decrement_ref_count);
    explicit MemoryPinImpl(std::shared_ptr<void> resource);

    void add_final_release_callback(std::function<void()> callback);

    void maybe_increment();
    void maybe_decrement();

    bool is_empty() const noexcept;

    // See experimental::MemoryPinMarkDeviceImmutable. Shared by every copy.
    void mark_device_immutable();
    bool is_device_immutable() const noexcept;

private:
    // State every copy of an impl shares. The last copy to go, on whichever thread, destroys it, which runs the
    // final-release callbacks exactly once; the shared_ptr release orders every add before that run.
    struct SharedState {
        ~SharedState();

        // Copies on different threads may add callbacks at the same time.
        std::mutex callbacks_mutex;
        std::vector<std::function<void()>> callbacks;
        std::atomic<bool> device_immutable{false};
    };

    // Created on first use for an impl constructed without one.
    SharedState& shared_state();

    std::function<void()> inc_;
    std::function<void()> dec_;
    std::shared_ptr<SharedState> shared_state_;
};

}  // namespace tt::tt_metal
