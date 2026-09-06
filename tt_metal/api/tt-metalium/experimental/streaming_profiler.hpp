// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <type_traits>

#include <tt-metalium/core_coord.hpp>

// Records from the streaming profiler, delivered to subscribers in batches:
//
//     auto handle = Subscribe("my-tool", [](const Batch<Channel::Zones | Channel::Stalls>& b) {
//         for (const Zone& z : b.zones) { use(z.site.name, z.core.coord, z.duration()); }
//         for (const Stall& s : b.stalls) { ... }
//         stalls += b.stall_count;
//     });
//     ...
//     Unsubscribe(handle);
//
// Timestamps are device clock ticks; a record's `clock` converts them to host time. `runtime_id` is the host id of
// the program running on the core, 0 if none was set. Strings live for the process; spans and clocks only for the
// callback they arrive in.
namespace tt::tt_metal::experimental::streaming_profiler {

enum class Risc : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4 };

struct SourceLocation {
    std::string_view file;
    uint32_t line = 0;
};

/**
 * @brief The marker's name and where it is written in kernel source.
 */
struct Site {
    std::string_view name;
    SourceLocation location;
};

/**
 * @brief The core a record came from, by the coordinate a program addresses it with.
 */
struct Core {
    CoreCoord coord;
    uint32_t chip_id = 0;
    Risc risc = Risc::BRISC;
};
static_assert(sizeof(Core) == 24);

/**
 * @brief Converts a chip's device timestamps to host time.
 *
 * If the chip could not be synchronized, `measured` is false, `frequency_ghz` is the nominal clock, and host_time()
 * is only meaningful for differences.
 */
struct Clock {
    double frequency_ghz = 0.0;  // device ticks per nanosecond
    uint64_t anchor_ticks = 0;
    int64_t anchor_host_ns = 0;  // std::chrono::steady_clock at `anchor_ticks`, in nanoseconds since its epoch
    uint32_t chip_id = 0;
    bool measured = false;

    std::chrono::nanoseconds duration(uint64_t from_ticks, uint64_t to_ticks) const {
        return std::chrono::nanoseconds(
            static_cast<int64_t>(static_cast<double>(static_cast<int64_t>(to_ticks - from_ticks)) / frequency_ghz));
    }
    std::chrono::steady_clock::time_point host_time(uint64_t ticks) const {
        return std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(anchor_host_ns) + duration(anchor_ticks, ticks));
    }
};
static_assert(sizeof(Clock) == 32);

/**
 * @brief One closed DeviceZoneScopedN scope. A core's zones arrive in close order: a nested child precedes its parent.
 */
struct Zone {
    Site site;
    Core core;
    std::reference_wrapper<const Clock> clock;  // the core's chip clock as of this batch
    uint64_t start_timestamp = 0;
    uint64_t end_timestamp = 0;
    uint32_t runtime_id = 0;

    std::chrono::nanoseconds duration() const { return clock.get().duration(start_timestamp, end_timestamp); }
    std::chrono::steady_clock::time_point start_time() const { return clock.get().host_time(start_timestamp); }
    std::chrono::steady_clock::time_point end_time() const { return clock.get().host_time(end_timestamp); }
};
static_assert(sizeof(Zone) == 96);

/**
 * @brief One DeviceTimestampedData marker with its payload, two 32-bit words per element, first word in the low half.
 */
struct TimestampedData {
    Site site;
    Core core;
    std::span<const uint64_t> payload;
    std::reference_wrapper<const Clock> clock;
    uint64_t timestamp = 0;
    uint32_t runtime_id = 0;

    std::chrono::steady_clock::time_point time() const { return clock.get().host_time(timestamp); }
};

/**
 * @brief One DeviceRecordEvent marker.
 */
struct Event {
    Site site;
    Core core;
    std::reference_wrapper<const Clock> clock;
    uint64_t timestamp = 0;
    uint32_t runtime_id = 0;

    std::chrono::steady_clock::time_point time() const { return clock.get().host_time(timestamp); }
};

/**
 * @brief An interval during which the profiler could not record a core's markers fast enough and the core waited.
 *        Zones on that core overlapping the interval include the wait.
 */
struct Stall {
    Core core;
    std::reference_wrapper<const Clock> clock;
    uint64_t start_timestamp = 0;
    uint64_t end_timestamp = 0;
    uint32_t runtime_id = 0;

    std::chrono::nanoseconds duration() const { return clock.get().duration(start_timestamp, end_timestamp); }
    std::chrono::steady_clock::time_point start_time() const { return clock.get().host_time(start_timestamp); }
    std::chrono::steady_clock::time_point end_time() const { return clock.get().host_time(end_timestamp); }
};

/**
 * @brief Record channels; combine with `|`.
 */
enum class Channel : uint32_t {
    Zones = 1u << 0,
    TimestampedData = 1u << 1,
    Events = 1u << 2,
    Stalls = 1u << 3,
    All = Zones | TimestampedData | Events | Stalls,
};
constexpr Channel operator|(Channel a, Channel b) {
    return static_cast<Channel>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

template <Channel K>
struct Batch;
using SubscriptionHandle = uint64_t;

// Implementation. Not part of the interface.
namespace detail {
constexpr bool has(Channel set, Channel channel) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(channel)) != 0;
}

template <bool>
struct ZonesMember {};
template <>
struct ZonesMember<true> {
    std::span<const Zone> zones;
};
template <bool>
struct TimestampedDataMember {};
template <>
struct TimestampedDataMember<true> {
    std::span<const TimestampedData> timestamped_data;
};
template <bool>
struct EventsMember {};
template <>
struct EventsMember<true> {
    std::span<const Event> events;
};
template <bool>
struct StallsMember {};
template <>
struct StallsMember<true> {
    std::span<const Stall> stalls;
};

SubscriptionHandle subscribe(
    std::string_view name, Channel channels, std::function<void(const Batch<Channel::All>&)> callback);

// The Batch<K> a callable's call operator takes.
template <typename F>
struct batch_of;
template <typename R, typename C, Channel K>
struct batch_of<R (C::*)(const Batch<K>&) const> {
    using type = Batch<K>;
};
template <typename R, typename C, Channel K>
struct batch_of<R (C::*)(const Batch<K>&)> {
    using type = Batch<K>;
};
template <typename R, Channel K>
struct batch_of<R (*)(const Batch<K>&)> {
    using type = Batch<K>;
};
template <typename F>
    requires requires { &F::operator(); }
struct batch_of<F> : batch_of<decltype(&F::operator())> {};
}  // namespace detail

/**
 * @brief One delivery: a span per channel in K (`zones`, `timestamped_data`, `events`, `stalls`), each oldest
 *        first, plus the facts every batch carries.
 */
template <Channel K>
struct Batch : detail::ZonesMember<detail::has(K, Channel::Zones)>,
               detail::TimestampedDataMember<detail::has(K, Channel::TimestampedData)>,
               detail::EventsMember<detail::has(K, Channel::Events)>,
               detail::StallsMember<detail::has(K, Channel::Stalls)> {
    static constexpr Channel channels = K;
    std::span<const Clock> clocks;  // one per captured chip; records point into it
    uint64_t dropped = 0;           // records lost since the previous batch because this subscriber fell behind
    uint64_t stall_count = 0;       // stalls in this batch on any core, whatever K is
};

namespace detail {
template <Channel K>
Batch<K> narrow(const Batch<Channel::All>& full) {
    Batch<K> b;
    b.clocks = full.clocks;
    b.dropped = full.dropped;
    b.stall_count = full.stall_count;
    if constexpr (has(K, Channel::Zones)) {
        b.zones = full.zones;
    }
    if constexpr (has(K, Channel::TimestampedData)) {
        b.timestamped_data = full.timestamped_data;
    }
    if constexpr (has(K, Channel::Events)) {
        b.events = full.events;
    }
    if constexpr (has(K, Channel::Stalls)) {
        b.stalls = full.stalls;
    }
    return b;
}
}  // namespace detail

/**
 * @brief Subscribe with a callable taking `const Batch<K>&`; K is the set of channels delivered.
 *
 * May be called at any time, before, during or between captures, and persists until Unsubscribe(). The callback
 * runs on its own thread, one call at a time; a subscriber that falls behind loses only its own records, counted in
 * Batch::dropped. Not to be called from inside a callback.
 *
 * @param name Appears in the profiler's logs and thread names; the nameless overload numbers the subscription.
 * @return A handle for Unsubscribe().
 */
template <typename F>
SubscriptionHandle Subscribe(std::string_view name, F callback) {
    using B = typename detail::batch_of<std::decay_t<F>>::type;
    constexpr Channel K = B::channels;
    return detail::subscribe(
        name, K, [cb = std::move(callback)](const Batch<Channel::All>& full) { cb(detail::narrow<K>(full)); });
}

template <typename F>
SubscriptionHandle Subscribe(F callback) {
    return Subscribe(std::string_view{}, std::move(callback));
}

/**
 * @brief Ends a subscription; returns once the callback can no longer run.
 */
void Unsubscribe(SubscriptionHandle handle);

/**
 * @brief True while a capture is running, from a MeshDevice opening with TT_METAL_STREAMING_PROFILER=1 until it closes.
 *        False on hardware the profiler does not support.
 */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
