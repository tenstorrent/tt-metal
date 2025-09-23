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
#include <thread>

namespace tt {
namespace utils {
bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);
const std::string& get_reports_dir();

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
