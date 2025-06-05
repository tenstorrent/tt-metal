// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>

#include <tt-logger/tt-logger.hpp>

namespace tt {
template <typename TimeUnit = std::chrono::nanoseconds>
struct ScopedTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TimeInstant = std::chrono::time_point<Clock, std::chrono::nanoseconds>;
    using Duration = std::chrono::duration<std::uint64_t, typename TimeUnit::period>;

    static std::string time_unit_to_string() {
        if constexpr (std::is_same_v<TimeUnit, std::chrono::milliseconds>) {
            return "ms";
        } else if constexpr (std::is_same_v<TimeUnit, std::chrono::microseconds>) {
            return "µs";
        } else if constexpr (std::is_same_v<TimeUnit, std::chrono::seconds>) {
            return "s";
        } else {
            return "ns";
        }
    }

    ScopedTimer(std::string name_, bool print_duration_ = true) : name(name_), print_duration(print_duration_) {
        this->start = std::chrono::time_point_cast<std::chrono::nanoseconds>(Clock::now());
    }

    ~ScopedTimer() {
        auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(Clock::now());
        this->elapsed = std::chrono::duration_cast<TimeUnit>(end - this->start);
        if (print_duration) {
            log_info(tt::LogTimer, "{} -- elapsed: {}{}", this->name, this->elapsed.count(), time_unit_to_string());
        }
    }

    std::string name;
    bool print_duration;
    TimeInstant start;
    Duration elapsed;
};

}  // namespace tt
