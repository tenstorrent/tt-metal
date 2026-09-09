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

// Records from the streaming profiler, delivered to registered callbacks in batches:
//
//     auto handle = RegisterCallback("my-tool", [](const Batch<RecordType::Zones>& b) {
//         for (const Zone& z : b.zones()) { use(z.site().name, z.core().logical, z.duration()); }
//     });
//     ...
//     UnregisterCallback(handle);
//
// A record is a value with no external references: copy it and keep it. Its timestamps are device clock ticks; it
// carries the chip's clock frequency and host offset that duration(), start_time() and time() apply. A
// TimestampedData carries its values inline, so it is as long as they make it. A batch's spans and ranges are valid
// only inside the callback.
namespace tt::tt_metal::streaming_profiler {
class Service;
}

namespace tt::tt_metal::experimental::streaming_profiler {

enum class Risc : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4 };

struct SourceLocation {
    std::string_view file;
    uint32_t line = 0;
};

/** @brief A marker's name and where it is written in kernel source. */
struct Site {
    std::string_view name;
    SourceLocation location;
};

/**
 * @brief The core a record came from. `logical` is the coordinate a program addresses it with; `physical` is its
 *        NoC 0 position on the die, which routing distance and harvesting act on.
 */
struct Core {
    CoreCoord logical;
    CoreCoord physical;
    uint32_t chip_id = 0;
    Risc risc = Risc::BRISC;
};

/** @brief Site name of a stall zone; it has no source location. */
inline constexpr std::string_view kStallZoneName = "PROFILER-STALL";

/** @brief Record types a callback receives; combine with `|`. */
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

// Implementation. Not part of the interface.
namespace detail {
constexpr bool has(RecordType set, RecordType type) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(type)) != 0;
}
constexpr bool covers(RecordType set, RecordType subset) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(subset)) == static_cast<uint32_t>(subset);
}

// A zone id is tu id << kZoneLocalBits | local id (hostdevcommon/profiler_zone_id.h).
inline constexpr uint32_t kZoneLocalBits = 14;
inline constexpr uint32_t kZoneTuCount = 1u << (27 - kZoneLocalBits);

// One translation unit's zone sites by local id, nullptr for an unnamed one; immutable once published.
struct SiteTu {
    std::span<const Site* const> sites;
};
// By tu id. A tu is published when its ELF loads, before any core can emit its ids, and replaced only by a superset.
extern std::atomic<const SiteTu*> g_site_tus[kZoneTuCount];
inline constexpr Site kUnnamedSite{};

inline const Site& site_of(uint32_t zone_id) {
    const SiteTu* tu = g_site_tus[zone_id >> kZoneLocalBits].load(std::memory_order_acquire);
    const uint32_t local = zone_id & ((1u << kZoneLocalBits) - 1u);
    const Site* s = tu != nullptr && local < tu->sites.size() ? tu->sites[local] : nullptr;
    return s != nullptr ? *s : kUnnamedSite;
}

CallbackHandle register_callback(std::string_view name, std::function<void(const Batch<RecordType::All>&)> callback);

// The Batch<K> a callable's call operator takes.
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

/** @brief The fields every record kind shares; Zone, TimestampedData and Event add their timestamps. */
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
    /** @brief Host id of the program running on the core, 0 if none was set. */
    uint32_t runtime_id() const { return runtime_id_; }
    /** @brief The chip's clock as measured for this record. */
    double frequency_ghz() const { return frequency_hz_ * 1e-9; }

protected:
    std::chrono::nanoseconds ticks_to_ns(uint64_t ticks) const {
        return std::chrono::nanoseconds(static_cast<int64_t>(static_cast<double>(ticks) * 1e9 / frequency_hz_));
    }
    // host_ns = (ticks + offset) / frequency: the offset is the host clock's origin in device cycles.
    std::chrono::steady_clock::time_point host_time(uint64_t ticks) const {
        const double cycles = static_cast<double>(static_cast<int64_t>(ticks) + offset_);
        return std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(static_cast<int64_t>(cycles * 1e9 / frequency_hz_)));
    }

    // In the order the decoder writes them. a_, b_ are a Zone's start and duration, an Event's timestamp, a
    // TimestampedData's timestamp and value count.
    uint64_t a_, b_;
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
 * @brief One closed DeviceZoneScopedN scope, or a stall (site name kStallZoneName): an interval during which the
 *        profiler could not record the core's markers fast enough and the core waited, included in the zones enclosing
 *        it. A core's zones arrive in close order: a nested child, a stall included, precedes its parent.
 */
class Zone : public Record {
public:
    uint64_t start_timestamp() const { return a_; }
    uint64_t end_timestamp() const { return a_ + b_; }
    std::chrono::nanoseconds duration() const { return ticks_to_ns(b_); }
    std::chrono::steady_clock::time_point start_time() const { return host_time(a_); }
    std::chrono::steady_clock::time_point end_time() const { return host_time(a_ + b_); }
};
static_assert(sizeof(Zone) == 48 && std::is_standard_layout_v<Zone>);

/**
 * @brief One DeviceTimestampedData marker with its values, which follow the record in memory: two 32-bit words per
 *        value, first word in the high half. The record is size_bytes() long, so a batch of them is a range, not a
 *        span; copy one with std::memcpy of size_bytes() to keep it.
 */
class TimestampedData : public Record {
public:
    uint64_t timestamp() const { return a_; }
    std::chrono::steady_clock::time_point time() const { return host_time(a_); }
    std::span<const uint64_t> payload() const {
        return std::span<const uint64_t>(reinterpret_cast<const uint64_t*>(this + 1), b_);
    }
    size_t size_bytes() const { return sizeof(TimestampedData) + b_ * sizeof(uint64_t); }

    /** @brief Steps over records packed back to back, each by its own size_bytes(). */
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
    uint64_t timestamp() const { return a_; }
    std::chrono::steady_clock::time_point time() const { return host_time(a_); }
};
static_assert(sizeof(Event) == 48 && std::is_standard_layout_v<Event>);

/**
 * @brief One delivery: the records of each type in K, oldest first. The spans and ranges are valid inside the
 *        callback; copy the records to keep them.
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
     * @brief Bytes of device output this callback missed since the previous batch because it fell behind; the records
     *        they held are gone uncounted, so this is a size, not a record count.
     */
    uint64_t dropped() const { return dropped_; }
    /**
     * @brief Stalls on any core since the previous batch, whatever K is: never fewer than the stall zones this batch
     *        holds, so zero means it holds none.
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
 * @brief Registers a callable taking `const Batch<K>&`; K selects the record types delivered.
 *
 * Allowed at any time, before, during or between captures; the callback persists until UnregisterCallback(). It runs
 * on its own thread, one call at a time, and a callback that falls behind loses only its own records
 * (Batch::dropped). Not to be called from inside a callback.
 *
 * @param name Appears in the profiler's logs and thread names; the nameless overload numbers the callback.
 * @return A handle for UnregisterCallback().
 */
template <typename F>
CallbackHandle RegisterCallback(std::string_view name, F callback) {
    using B = typename detail::batch_of<std::decay_t<F>>::type;
    return detail::register_callback(
        name, [cb = std::move(callback)](const Batch<RecordType::All>& full) { cb(B(full)); });
}

template <typename F>
CallbackHandle RegisterCallback(F callback) {
    return RegisterCallback(std::string_view{}, std::move(callback));
}

/** @brief Removes a callback; returns once it can no longer run. */
void UnregisterCallback(CallbackHandle handle);

/**
 * @brief True while a capture is running, from a MeshDevice opening with TT_METAL_STREAMING_PROFILER=1 until it
 *        closes.
 */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
