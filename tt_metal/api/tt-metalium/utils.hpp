// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <chrono>
#include <stdexcept>
#include <future>

namespace tt {
namespace utils {
bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);
const std::string& get_reports_dir();

// Cancellable timeout wrapper: invokes on_timeout() before throwing and waits for task to exit
// Please note that the FuncBody is going to loop until the FuncWait returns false.
template <typename FuncBody, typename FuncWait, typename OnTimeout, typename... Args>
auto loop_and_wait_with_timeout(
    FuncBody&& func_body,
    FuncWait&& wait_condition,
    OnTimeout&& on_timeout,
    std::chrono::duration<float> timeout_duration,
    Args&&... args) {
    if (timeout_duration.count() > 0.0f) {
        auto start_time = std::chrono::high_resolution_clock::now();

        do {
            func_body(args...);
            if (wait_condition(args...)) {
                // If somehow finished up the operation, we don't need to yield
                std::this_thread::yield();
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(current_time - start_time).count();

            if (elapsed >= timeout_duration.count()) {
                on_timeout();
                break;
            }
        } while (wait_condition(args...));
    } else {
        do {
            func_body(args...);
        } while (wait_condition(args...));
    }
}

// Ripped out of boost for std::size_t so as to not pull in bulky boost dependencies
template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename E, std::enable_if_t<std::is_enum<E>::value, bool> = true>
auto underlying_type(const E& e) {
    return static_cast<typename std::underlying_type<E>::type>(e);
}
}  // namespace utils
}  // namespace tt
