// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <thread>
#include <type_traits>

namespace tt::tt_metal {

template <typename Predicate, typename Progress>
bool poll_until(
    Predicate predicate,
    Progress progress,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval) {
    static_assert(std::is_same_v<std::invoke_result_t<Predicate>, bool>);
    static_assert(std::is_invocable_v<Progress>);

    const auto start = std::chrono::steady_clock::now();
    while (!predicate()) {
        if (std::chrono::steady_clock::now() - start >= timeout) {
            return false;
        }
        progress();
        std::this_thread::sleep_for(poll_interval);
    }
    return true;
}

}  // namespace tt::tt_metal
