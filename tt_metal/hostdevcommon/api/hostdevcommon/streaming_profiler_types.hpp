// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compiled as C++17 for the device and C++20 for the host.
#pragma once

#include <cstdint>
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include <cstddef>
#else
#include <chrono>
#include <cmath>
#include <string_view>
#endif

namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief A string field or value: written "..."_s in a kernel, and read on the host as a std::string_view that lasts
 * for the process. */
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
struct String {
    String() = default;
    // Only "..."_s puts the text where the host can read it, so a plain literal is a compile error that says so.
    template <std::size_t N>
    constexpr String(const char (&)[N]) {
        static_assert(N == 0, "strings sent with DevicePublish need an _s suffix: \"text\"_s");
    }
    const char* text = nullptr;
};
#else
using String = std::string_view;
#endif

/** @brief A DeviceZoneScopedN zone's metadata. */
struct ZoneMetadata {
    String name;
};
/** @brief A DeviceRecordEvent marker's metadata. */
struct EventMetadata {
    String name;
};
/** @brief A DeviceTimestampedData marker's metadata. */
struct TimestampedDataMetadata {
    String name;
};

/**
 * @brief A time on a core's wall clock: a kernel takes it with Timestamp::now(), and the host reads it on steady_clock,
 * in host reference cycles or in the core's own clock cycles.
 */
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
class Timestamp {
public:
    Timestamp() = default;
    /** @brief The core's wall clock now. */
    static Timestamp now();

private:
    uint64_t ticks_ = 0;
};
#else
class Timestamp {
public:
    /** @brief The timestamp on steady_clock. */
    std::chrono::steady_clock::time_point time() const {
        const TscToSteadyLine& line = tsc_to_steady_line_;
        if (tsc_ < line.from || tsc_ > line.to) {
            return refill_tsc_to_steady_line(tsc_);
        }
        return line.at(tsc_);
    }
    /** @brief The timestamp in host reference cycles (the x86 time-stamp counter). */
    int64_t host_cycles() const { return tsc_; }
    /**
     * @brief The timestamp in the core's own clock cycles.
     *
     * Not comparable across cores.
     */
    uint64_t device_cycles() const { return device_cycles_; }

private:
    struct TscToSteadyLine {
        int64_t from = 0, to = -1;
        int64_t origin = 0, base_ns = 0;
        double value = 0.0, slope = 0.0;
        std::chrono::steady_clock::time_point at(int64_t tsc) const {
            return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(
                base_ns + static_cast<int64_t>(std::nearbyint(value + slope * static_cast<double>(tsc - origin)))));
        }
    };
    static std::chrono::steady_clock::time_point refill_tsc_to_steady_line(int64_t tsc);
    static constinit thread_local TscToSteadyLine tsc_to_steady_line_;

    uint64_t device_cycles_ = 0;
    int64_t tsc_ = 0;
};
#endif

/** @brief A start and an end Timestamp. */
struct Interval {
    Timestamp start, end;

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
    /** @brief The time from start to end. */
    std::chrono::nanoseconds duration() const { return end.time() - start.time(); }
    /** @brief The core's average clock frequency from start to end. */
    double frequency_ghz() const {
        return static_cast<double>(end.device_cycles() - start.device_cycles()) /
               static_cast<double>(duration().count());
    }
#endif
};

}  // namespace tt::tt_metal::experimental::streaming_profiler
