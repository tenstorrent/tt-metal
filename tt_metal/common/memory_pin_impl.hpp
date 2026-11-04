// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
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

private:
    struct FinalReleaseState {
        std::vector<std::function<void()>> callbacks;
        bool ran = false;
    };

    void maybe_run_final_release_callbacks();

    std::function<void()> inc_;
    std::function<void()> dec_;
    std::shared_ptr<FinalReleaseState> final_release_state_;
};

}  // namespace tt::tt_metal
