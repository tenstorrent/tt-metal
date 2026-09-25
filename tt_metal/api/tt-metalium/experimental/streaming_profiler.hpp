// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <concepts>
#include <ratio>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>

// Streaming profiler host API: a callback registered here receives the records device kernels emit.
//
// This API is experimental and may change or be removed without notice.
//
//     auto handle = RegisterCallback("my-tool", [](const Batch<RecordType::Zones>& b) {
//         for (const Zone& z : b.zones()) {
//             use(z.site().name, z.core().logical, z.duration());
//         }
//     });
//     ...
//     UnregisterCallback(handle);
//
namespace tt::tt_metal::streaming_profiler {
class Service;
}

namespace tt::tt_metal::experimental::streaming_profiler {

enum class Risc : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4, ERISC0 = 5, ERISC1 = 6 };

struct SourceLocation {
    std::string_view file;  // Valid for the lifetime of the process.
    uint32_t line = 0;
};

/** @brief A marker's name and where it is written in kernel source code. */
struct MarkerSite {
    std::string_view name;  // Valid for the lifetime of the process.
    SourceLocation location;
};

/**
 * @brief The core a record came from.
 *
 * `logical` is the coordinate a program addresses it with, in the Ethernet cores' own logical space when `risc` is
 * ERISC0 or ERISC1; `physical` is its NoC 0 position on the die.
 */
struct Core {
    CoreCoord logical;
    CoreCoord physical;
    ChipId chip_id = 0;
    Risc risc = Risc::BRISC;
};

/** @brief Site name of a stall zone. */
inline constexpr std::string_view STALL_ZONE_NAME = "PROFILER-STALL";

/** @brief Record types a callback can subscribe to; combine with `|`. */
enum class RecordType : uint32_t {
    Zones = 1u << 0,
    TimestampedData = 1u << 1,
    Events = 1u << 2,
    All = Zones | TimestampedData | Events,
};
constexpr RecordType operator|(RecordType a, RecordType b) {
    return static_cast<RecordType>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

template <RecordType K>
class Batch;

enum class CallbackHandle : uint64_t {};

// Implementation detail.
namespace detail {
constexpr bool has(RecordType set, RecordType type) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(type)) != 0;
}
constexpr bool covers(RecordType set, RecordType subset) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(subset)) == static_cast<uint32_t>(subset);
}

inline constexpr uint32_t ZONE_ID_BITS = 27;
inline constexpr uint32_t ZONE_LOCAL_BITS = 14;
inline constexpr uint32_t ZONE_TU_COUNT = 1u << (ZONE_ID_BITS - ZONE_LOCAL_BITS);

struct SiteTu {
    std::span<const MarkerSite* const> sites;
};
struct SiteRegistry {
    static std::atomic<const SiteTu*> tus[ZONE_TU_COUNT];
};
inline constexpr MarkerSite UNNAMED_SITE{};

inline const MarkerSite& site_of(uint32_t zone_id) {
    const SiteTu* tu = SiteRegistry::tus[zone_id >> ZONE_LOCAL_BITS].load(std::memory_order_acquire);
    const uint32_t local = zone_id & ((1u << ZONE_LOCAL_BITS) - 1u);
    const MarkerSite* s = tu != nullptr && local < tu->sites.size() ? tu->sites[local] : nullptr;
    return s != nullptr ? *s : UNNAMED_SITE;
}

CallbackHandle register_callback(std::string name, std::function<void(const Batch<RecordType::All>&)> callback);

template <typename Signature>
inline constexpr RecordType batch_arg = RecordType{};
template <typename R, RecordType K>
inline constexpr RecordType batch_arg<std::function<R(const Batch<K>&)>> = K;
template <typename R, RecordType K>
inline constexpr RecordType batch_arg<std::function<R(Batch<K>)>> = K;

template <typename F>
constexpr RecordType accepted_batch() {
    using G = std::remove_reference_t<std::unwrap_ref_decay_t<F>>;
    if constexpr (requires { std::function{std::declval<G>()}; }) {
        return batch_arg<decltype(std::function{std::declval<G>()})>;
    } else {
        return RecordType{};
    }
}

int64_t host_to_steady_ns(int64_t host) noexcept;
}  // namespace detail

/**
 * @brief The clock every record's host time is in: the host's time-stamp counter, in tenths of a nanosecond.
 *
 * steady_time() converts to the OS clock; it is std::chrono::clock_cast<std::chrono::steady_clock> for this clock,
 * spelled out because the C++20 chrono conversions need libstdc++ 13.
 */
struct host_clock {
    using rep = int64_t;
    using period = std::ratio<1, 10'000'000'000>;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<host_clock>;
    static constexpr bool is_steady = true;
    /** @brief The clock now. */
    static time_point now() noexcept;
    /** @brief The time-stamp counter's count at a time point. */
    static int64_t tsc(time_point t) noexcept;
    /** @brief The time point of a time-stamp counter count. */
    static time_point from_tsc(int64_t ticks) noexcept;
};

/** @brief A host_clock time point on std::chrono::steady_clock. */
inline std::chrono::steady_clock::time_point steady_time(host_clock::time_point t) noexcept {
    return std::chrono::steady_clock::time_point(
        std::chrono::nanoseconds(detail::host_to_steady_ns(t.time_since_epoch().count())));
}

/** @brief Base class of every record: its site, core, program id and clock. */
class Record {
public:
    /** @brief The marker this record came from. */
    const MarkerSite& site() const { return detail::site_of(zone_id_); }
    /** @brief The core that emitted the record. */
    Core core() const {
        return Core{
            .logical = CoreCoord(logical_x_, logical_y_),
            .physical = CoreCoord(physical_x_, physical_y_),
            .chip_id = chip_id_,
            .risc = static_cast<Risc>(risc_)};
    }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const { return runtime_id_; }

protected:
    friend class tt::tt_metal::streaming_profiler::Service;  // writes host_time_ and host_end_ at release
    uint64_t timestamp_;
    union {
        uint64_t duration_;     // Zone
        uint64_t value_count_;  // TimestampedData
    };
    uint32_t zone_id_;
    uint32_t runtime_id_;
    uint16_t logical_x_, logical_y_, physical_x_, physical_y_;
    uint16_t chip_id_;
    uint8_t risc_;
    uint32_t reserved_;
    // The record's instant (a zone's start) and a zone's end on host_clock, written by the service when the batch is
    // released to its consumer. Until then host_time_ holds the lane's tile offset, which takes its ticks into the
    // chip's eth wall domain, the input of that placement.
    int64_t host_time_;
    int64_t host_end_;
};
static_assert(sizeof(Record) == 56);

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone with site name
 * STALL_ZONE_NAME.
 */
class Zone : public Record {
public:
    /** @brief Start of the zone in device clock ticks. */
    uint64_t start_timestamp() const { return timestamp_; }
    /** @brief End of the zone in device clock ticks. */
    uint64_t end_timestamp() const { return timestamp_ + duration_; }
    /** @brief Start of the zone on the host clock. */
    host_clock::time_point start_time() const { return host_clock::time_point(host_clock::duration(host_time_)); }
    /** @brief End of the zone on the host clock. */
    host_clock::time_point end_time() const { return host_clock::time_point(host_clock::duration(host_end_)); }
    /** @brief Start and end of the zone on the host clock. */
    std::pair<host_clock::time_point, host_clock::time_point> host_span() const { return {start_time(), end_time()}; }
    /** @brief Length of the zone on the host clock. */
    std::chrono::nanoseconds duration() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time() - start_time());
    }
    /** @brief The chip's clock frequency over the zone in GHz: its device ticks over its host length. */
    double frequency_ghz() const {
        const int64_t host = host_end_ - host_time_;
        return host > 0 ? static_cast<double>(duration_) * 10.0 / static_cast<double>(host) : 0.0;
    }
};
static_assert(sizeof(Zone) == 56 && std::is_standard_layout_v<Zone>);

/**
 * @brief One DeviceTimestampedData marker with its values.
 *
 * Variable-length: to keep one past the callback, memcpy its size_bytes() bytes.
 */
class TimestampedData : public Record {
public:
    TimestampedData() = default;
    TimestampedData(const TimestampedData&) = delete;
    TimestampedData& operator=(const TimestampedData&) = delete;
    /** @brief When the marker was recorded, in device clock ticks. */
    uint64_t timestamp() const { return timestamp_; }
    /** @brief When the marker was recorded, on the host clock. */
    host_clock::time_point time() const { return host_clock::time_point(host_clock::duration(host_time_)); }
    /** @brief The marker's values. */
    std::span<const uint64_t> payload() const {
        return std::span<const uint64_t>(reinterpret_cast<const uint64_t*>(this + 1), value_count_);
    }
    /** @brief Size of the record including its values. */
    size_t size_bytes() const { return sizeof(TimestampedData) + value_count_ * sizeof(uint64_t); }

    /** @brief Iterates the TimestampedData records of a batch. */
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = TimestampedData;
        using difference_type = std::ptrdiff_t;

        iterator() = default;
        explicit iterator(const std::byte* p) : p_(p) {}
        const TimestampedData& operator*() const { return *reinterpret_cast<const TimestampedData*>(p_); }
        const TimestampedData* operator->() const { return reinterpret_cast<const TimestampedData*>(p_); }
        iterator& operator++() {
            p_ += (**this).size_bytes();
            return *this;
        }
        iterator operator++(int) {
            iterator old = *this;
            ++*this;
            return old;
        }
        bool operator==(const iterator&) const = default;

    private:
        const std::byte* p_ = nullptr;
    };
};
static_assert(sizeof(TimestampedData) == 56 && std::is_standard_layout_v<TimestampedData>);

/** @brief One DeviceRecordEvent marker. */
class Event : public Record {
public:
    /** @brief When the marker was recorded, in device clock ticks. */
    uint64_t timestamp() const { return timestamp_; }
    /** @brief When the marker was recorded, on the host clock. */
    host_clock::time_point time() const { return host_clock::time_point(host_clock::duration(host_time_)); }
};
static_assert(sizeof(Event) == 56 && std::is_standard_layout_v<Event>);

/**
 * @brief The records of each type in K, in the order the device emitted them.
 *
 * The spans and ranges are valid inside the callback; copy the records to keep them.
 */
template <RecordType K>
class Batch {
public:
    static constexpr RecordType types = K;

    Batch() = default;
    template <RecordType K2>
        requires(detail::covers(K2, K))
    explicit Batch(const Batch<K2>& full) :
        zones_(full.zones_),
        timestamped_data_(full.timestamped_data_),
        events_(full.events_),
        dropped_(full.dropped_),
        stall_count_(full.stall_count_) {}

    std::span<const Zone> zones() const
        requires(detail::has(K, RecordType::Zones))
    {
        return zones_;
    }
    std::ranges::subrange<TimestampedData::iterator> timestamped_data() const
        requires(detail::has(K, RecordType::TimestampedData))
    {
        return timestamped_data_;
    }
    std::span<const Event> events() const
        requires(detail::has(K, RecordType::Events))
    {
        return events_;
    }
    /**
     * @brief Bytes of device output lost since this callback last ran; nonzero if the callback could not keep up with
     *        incoming data.
     *
     * Raising TT_METAL_STREAMING_PROFILER_FIFO_MB lets a callback fall further behind before it loses data.
     */
    uint64_t dropped_bytes() const { return dropped_; }
    /**
     * @brief Stalls on any core since the previous batch: times a core waited for the profiler to drain its records.
     */
    uint64_t stall_count() const { return stall_count_; }

private:
    template <RecordType>
    friend class Batch;
    friend class tt::tt_metal::streaming_profiler::Service;
    std::span<const Zone> zones_;
    std::ranges::subrange<TimestampedData::iterator> timestamped_data_;
    std::span<const Event> events_;
    uint64_t dropped_ = 0;
    uint64_t stall_count_ = 0;
};

/**
 * @brief Registers a callback to be invoked when streaming profiler records arrive from a device.
 *
 * The callable takes `const Batch<K>&`; K selects the record types delivered. Multiple callbacks can be registered;
 * each runs on its own thread, one invocation at a time. If a callback shares a resource with other callbacks, access
 * it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep up with incoming data miss records;
 * this is reported by Batch::dropped_bytes. May be called before, during or between captures.
 *
 * @param name Appears in the profiler's logs and thread names.
 * @return A handle that can be passed to UnregisterCallback() to remove the callback.
 */
template <typename F>
[[nodiscard]] CallbackHandle RegisterCallback(std::string name, F callback) {
    constexpr RecordType K = detail::accepted_batch<F>();
    static_assert(K != RecordType{}, "the callback must take one Batch<K> parameter");
    static_assert(std::is_copy_constructible_v<F>, "the callback must be copy-constructible");
    return detail::register_callback(
        std::move(name),
        [cb = std::move(callback)](const Batch<RecordType::All>& full) mutable { cb(Batch<K>(full)); });
}

/**
 * @brief Registers a callback, under a generated name, to be invoked when streaming profiler records arrive from a
 *        device.
 *
 * The callable takes `const Batch<K>&`; K selects the record types delivered. Multiple callbacks can be registered;
 * each runs on its own thread, one invocation at a time. If a callback shares a resource with other callbacks, access
 * it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep up with incoming data miss records;
 * this is reported by Batch::dropped_bytes. May be called before, during or between captures.
 *
 * @return A handle that can be passed to UnregisterCallback() to remove the callback.
 */
template <typename F>
[[nodiscard]] CallbackHandle RegisterCallback(F callback) {
    return RegisterCallback(std::string{}, std::move(callback));
}

/**
 * @brief Unregisters a previously registered callback by its handle.
 *
 * Blocks until any in-flight invocation of that callback has completed.
 */
void UnregisterCallback(CallbackHandle handle);

/**
 * @brief Returns true if the streaming profiler is currently running on at least one chip.
 */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
