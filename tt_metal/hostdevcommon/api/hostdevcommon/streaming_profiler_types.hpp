// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compiled as C++17 for the device and C++20 for the host.
#pragma once

#include <cstddef>
#include <cstdint>

// Defined when compiling kernels or firmware. String, Timestamp and Interval are then what a kernel sends; the host reads
// the same fields as a std::string_view and host time.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#define TT_DEVICE_COMPILE
#endif

#ifdef TT_DEVICE_COMPILE

namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief A string field of metadata: written "..."_s in a kernel, and read on the host as a std::string_view that
 * lasts for the process. */
struct String {
    String() = default;
    // Only "..."_s puts the text where the host can read it, so a plain literal is a compile error that says so.
    template <std::size_t N>
    constexpr String(const char (&)[N]) {
        static_assert(N == 0, "strings sent with DevicePublish need an _s suffix: \"text\"_s");
    }
    // The 64-bit hash of its text, which "..."_s keeps in the image, where the host looks it up by this hash.
    uint64_t hash = 0;
};

/**
 * @brief A time on a core's wall clock: a kernel takes it with Timestamp::now(), and the host reads it on steady_clock,
 * in host reference cycles or in the core's own clock cycles.
 */
class Timestamp {
public:
    Timestamp() = default;
    /** @brief The core's wall clock now. */
    static Timestamp now();

private:
    explicit Timestamp(uint64_t ticks) : ticks_(ticks) {}
    uint64_t ticks_ = 0;
};

/** @brief A start and an end Timestamp. */
struct Interval {
    Timestamp start, end;
};

}  // namespace tt::tt_metal::experimental::streaming_profiler

#else

#include <chrono>
#include <cmath>
#include <string_view>

namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief A string field of metadata: written "..."_s in a kernel, and read on the host as a std::string_view that
 * lasts for the process. */
using String = std::string_view;

/**
 * @brief A time on a core's wall clock: a kernel takes it with Timestamp::now(), and the host reads it on steady_clock,
 * in host reference cycles or in the core's own clock cycles.
 */
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

/** @brief A start and an end Timestamp. */
struct Interval {
    Timestamp start, end;

    /** @brief The time from start to end. */
    std::chrono::nanoseconds duration() const { return end.time() - start.time(); }
    /** @brief The core's average clock frequency from start to end. */
    double frequency_ghz() const {
        return static_cast<double>(end.device_cycles() - start.device_cycles()) /
               static_cast<double>(duration().count());
    }
};

}  // namespace tt::tt_metal::experimental::streaming_profiler

#endif

namespace tt::tt_metal::experimental::streaming_profiler {

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
/** @brief A stall's metadata, which has none. */
struct StallMetadata {};

namespace detail {
// The metadata of a record sent without any, `DevicePublish<>(payload)`, which the host's Record<Payload, void>
// matches.
struct NoMetadata {};
}  // namespace detail

}  // namespace tt::tt_metal::experimental::streaming_profiler
