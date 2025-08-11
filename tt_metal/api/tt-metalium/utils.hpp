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

float get_timeout_seconds_for_operations();

// Cancellable timeout wrapper: invokes on_timeout() before throwing and waits for task to exit
template <typename FuncBody, typename FuncWait, typename OnTimeout, typename... Args>
auto wait_with_timeout(
    FuncBody&& func_body, FuncWait&& wait_condition, OnTimeout&& on_timeout, float timeout_seconds, Args&&... args) {
    auto start_time = std::chrono::high_resolution_clock::now();

    do {
        func_body(std::forward<Args>(args)...);
        asm("pause");  // Busy wait

        if (timeout_seconds > 0.0f) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(current_time - start_time).count();

            if (elapsed >= timeout_seconds) {
                on_timeout();
                break;
            }
        }
    } while (wait_condition(std::forward<Args>(args)...));
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
