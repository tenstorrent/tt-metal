// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>

#include <tt-metalium/core_coord.hpp>

// Streaming profiler host API: a callback registered here receives the records device kernels emit.
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

enum class Risc : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4 };

struct SourceLocation {
    std::string_view file;
    uint32_t line = 0;
};

/** @brief A marker's name and where it is written in kernel source code. */
struct Site {
    std::string_view name;
    SourceLocation location;
};

/**
 * @brief The core a record came from.
 *
 * `logical` is the coordinate a program addresses it with; `physical` is its NoC 0 position on the die.
 */
struct Core {
    CoreCoord logical;
    CoreCoord physical;
    uint32_t chip_id = 0;
    Risc risc = Risc::BRISC;
};

/** @brief Site name of a stall zone. */
inline constexpr std::string_view kStallZoneName = "PROFILER-STALL";

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

using CallbackHandle = uint64_t;

// Implementation detail.
namespace detail {
constexpr bool has(RecordType set, RecordType type) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(type)) != 0;
}
constexpr bool covers(RecordType set, RecordType subset) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(subset)) == static_cast<uint32_t>(subset);
}

inline constexpr uint32_t kZoneLocalBits = 14;
inline constexpr uint32_t kZoneTuCount = 1u << (27 - kZoneLocalBits);

struct SiteTu {
    std::span<const Site* const> sites;
};
extern std::atomic<const SiteTu*> g_site_tus[kZoneTuCount];
inline constexpr Site kUnnamedSite{};

inline const Site& site_of(uint32_t zone_id) {
    const SiteTu* tu = g_site_tus[zone_id >> kZoneLocalBits].load(std::memory_order_acquire);
    const uint32_t local = zone_id & ((1u << kZoneLocalBits) - 1u);
    const Site* s = tu != nullptr && local < tu->sites.size() ? tu->sites[local] : nullptr;
    return s != nullptr ? *s : kUnnamedSite;
}

CallbackHandle register_callback(std::string_view name, std::function<void(const Batch<RecordType::All>&)> callback);

template <typename F>
struct batch_of;
template <typename R, typename C, RecordType K>
struct batch_of<R (C::*)(const Batch<K>&) const> {
    using type = Batch<K>;
};
template <typename R, typename C, RecordType K>
struct batch_of<R (C::*)(const Batch<K>&)> {
    using type = Batch<K>;
};
template <typename R, RecordType K>
struct batch_of<R (*)(const Batch<K>&)> {
    using type = Batch<K>;
};
template <typename F>
    requires requires { &F::operator(); }
struct batch_of<F> : batch_of<decltype(&F::operator())> {};
}  // namespace detail

/** @brief Base class of every record: its site, core, program id and clock. */
namespace detail {
// This chip's time-indexed device<->device sync correction to the record's baked anchor, in nanoseconds: 0 until
// the d2d sync publishes one, so a record converts exactly as before by default. Defined in the library.
int64_t sync_correction_ns(uint16_t chip_id, int64_t host_ns) noexcept;
}  // namespace detail

class Record {
public:
    const Site& site() const { return detail::site_of(zone_id_); }
    Core core() const {
        return Core{
            .logical = CoreCoord(logical_x_, logical_y_),
            .physical = CoreCoord(physical_x_, physical_y_),
            .chip_id = chip_id_,
            .risc = static_cast<Risc>(risc_)};
    }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const { return runtime_id_; }
    /** @brief The chip's clock frequency in GHz. */
    double frequency_ghz() const { return frequency_hz_ * 1e-9; }

protected:
    std::chrono::nanoseconds ticks_to_ns(uint64_t ticks) const {
        return std::chrono::nanoseconds(static_cast<int64_t>(static_cast<double>(ticks) * 1e9 / frequency_hz_));
    }
    std::chrono::steady_clock::time_point host_time(uint64_t ticks) const {
        const double cycles = static_cast<double>(static_cast<int64_t>(ticks) + offset_);
        // The baked scalar anchor (this record's own wall-clock domain -> host ns), composed with the chip's
        // time-indexed d2d sync term. The term is keyed by HOST TIME, not wall ticks: eth and worker tiles keep
        // different wall-clock totals (per-card duty cycle), so a correction measured on the eth core is applied
        // to a worker zone through their common host reference -- each maps its own ticks to host, then looks up.
        const int64_t base_ns = static_cast<int64_t>(cycles * 1e9 / frequency_hz_);
        return std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(base_ns + detail::sync_correction_ns(chip_id_, base_ns)));
    }

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
    uint32_t frequency_hz_;
    int64_t offset_;
};
static_assert(sizeof(Record) == 48);

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone with site name
 * kStallZoneName.
 */
class Zone : public Record {
public:
    /** @brief Start of the zone in device clock ticks. */
    uint64_t start_timestamp() const { return timestamp_; }
    /** @brief End of the zone in device clock ticks. */
    uint64_t end_timestamp() const { return timestamp_ + duration_; }
    /** @brief Length of the zone. */
    std::chrono::nanoseconds duration() const { return ticks_to_ns(duration_); }
    /** @brief Start of the zone on the host clock. */
    std::chrono::steady_clock::time_point start_time() const { return host_time(timestamp_); }
    /** @brief End of the zone on the host clock. */
    std::chrono::steady_clock::time_point end_time() const { return host_time(timestamp_ + duration_); }
};
static_assert(sizeof(Zone) == 48 && std::is_standard_layout_v<Zone>);

/**
 * @brief One DeviceTimestampedData marker with its values.
 *
 * Variable-length: size_bytes() long, so copy it with memcpy, not by assignment.
 */
class TimestampedData : public Record {
public:
    /** @brief When the marker was recorded, in device clock ticks. */
    uint64_t timestamp() const { return timestamp_; }
    /** @brief When the marker was recorded, on the host clock. */
    std::chrono::steady_clock::time_point time() const { return host_time(timestamp_); }
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
static_assert(sizeof(TimestampedData) == 48 && std::is_standard_layout_v<TimestampedData>);

/** @brief One DeviceRecordEvent marker. */
class Event : public Record {
public:
    /** @brief When the marker was recorded, in device clock ticks. */
    uint64_t timestamp() const { return timestamp_; }
    /** @brief When the marker was recorded, on the host clock. */
    std::chrono::steady_clock::time_point time() const { return host_time(timestamp_); }
};
static_assert(sizeof(Event) == 48 && std::is_standard_layout_v<Event>);

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
    uint64_t dropped() const { return dropped_; }
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
 * this is reported by Batch::dropped. May be called before, during or between captures.
 *
 * @param name Appears in the profiler's logs and thread names.
 * @return A handle that can be passed to UnregisterCallback() to remove the callback.
 */
template <typename F>
CallbackHandle RegisterCallback(std::string_view name, F callback) {
    using B = typename detail::batch_of<std::decay_t<F>>::type;
    return detail::register_callback(
        name, [cb = std::move(callback)](const Batch<RecordType::All>& full) { cb(B(full)); });
}

/**
 * @brief Registers a callback, under a generated name, to be invoked when streaming profiler records arrive from a
 *        device.
 *
 * The callable takes `const Batch<K>&`; K selects the record types delivered. Multiple callbacks can be registered;
 * each runs on its own thread, one invocation at a time. If a callback shares a resource with other callbacks, access
 * it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep up with incoming data miss records;
 * this is reported by Batch::dropped. May be called before, during or between captures.
 *
 * @return A handle that can be passed to UnregisterCallback() to remove the callback.
 */
template <typename F>
CallbackHandle RegisterCallback(F callback) {
    return RegisterCallback(std::string_view{}, std::move(callback));
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
