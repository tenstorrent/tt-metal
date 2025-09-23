// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <thread>

namespace tt {
namespace tt_metal {
namespace dispatch {
namespace utils {

// Cancellable timeout wrapper: invokes on_timeout() before throwing and waits for task to exit
// Please note that the FuncBody is going to loop until the FuncWait returns false.
template <typename FuncBody, typename FuncWait, typename OnTimeout>
void loop_and_wait_with_timeout(
    const FuncBody& func_body,
    const FuncWait& wait_condition,
    const OnTimeout& on_timeout,
    std::chrono::duration<float> timeout_duration) {
    if (timeout_duration.count() > 0.0f) {
        auto start_time = std::chrono::high_resolution_clock::now();

        do {
            func_body();
            if (wait_condition()) {
                // If somehow finished up the operation, we don't need to yield
                std::this_thread::yield();
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(current_time - start_time).count();

            if (elapsed >= timeout_duration.count()) {
                on_timeout();
                break;
            }
        } while (wait_condition());
    } else {
        do {
            func_body();
        } while (wait_condition());
    }
}

}  // namespace utils
}  // namespace dispatch
}  // namespace tt_metal
}  // namespace tt