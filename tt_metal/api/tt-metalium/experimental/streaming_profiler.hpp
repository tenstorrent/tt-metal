// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
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
    std::string_view file;  // Valid for the lifetime of the process.
    uint32_t line = 0;
};

/** @brief A marker's name and where it is written in kernel source code. */
struct Site {
    std::string_view name;  // Valid for the lifetime of the process.
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

enum class CallbackHandle : uint64_t {};

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

CallbackHandle register_callback(std::string name, std::function<void(const Batch<RecordType::All>&)> callback);

template <typename T>
inline constexpr bool is_batch = false;
template <RecordType K>
inline constexpr bool is_batch<Batch<K>> = true;
template <typename R, typename A>
auto param(R (*)(A)) -> A;
template <typename R, typename A>
auto param(R (*)(A) noexcept) -> A;
template <typename R, typename C, typename A>
auto param(R (C::*)(A)) -> A;
template <typename R, typename C, typename A>
auto param(R (C::*)(A) const) -> A;
template <typename R, typename C, typename A>
auto param(R (C::*)(A) noexcept) -> A;
template <typename R, typename C, typename A>
auto param(R (C::*)(A) const noexcept) -> A;
template <typename F>
auto callback_param(int) -> decltype(param(&F::operator()));
template <typename F>
    requires std::is_pointer_v<F>
auto callback_param(long) -> decltype(param(std::declval<F>()));
template <typename F>
using callback_param_t = decltype(callback_param<F>(0));
template <typename F>
concept batch_callback = is_batch<std::remove_cvref_t<callback_param_t<F>>>;

int64_t sync_correction_ns(uint16_t chip_id, int64_t host_ns) noexcept;
void sync_correction_span_ns(uint16_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept;
}  // namespace detail

/** @brief Base class of every record: its site, core, program id and clock. */
class Record {
public:
    /** @brief The marker this record came from. */
    const Site& site() const { return detail::site_of(zone_id_); }
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
    /** @brief The chip's clock frequency in GHz. */
    double frequency_ghz() const { return frequency_hz_ * 1e-9; }
    /** @brief Host ns of a device tick from the chip's static anchor alone, before the d2d correction. */
    int64_t base_ns(uint64_t ticks) const {
        const double cycles = static_cast<double>(static_cast<int64_t>(ticks) + offset_);
        return static_cast<int64_t>(cycles * 1e9 / frequency_hz_);
    }

protected:
    std::chrono::nanoseconds ticks_to_ns(uint64_t ticks) const {
        return std::chrono::nanoseconds(static_cast<int64_t>(static_cast<double>(ticks) * 1e9 / frequency_hz_));
    }
    std::chrono::steady_clock::time_point host_time(uint64_t ticks) const {
        // The baked scalar anchor composed with the chip's time-indexed d2d sync term. The term is keyed by HOST
        // TIME, not wall ticks: eth and worker tiles keep different wall-clock totals (per-card duty cycle), so a
        // correction measured on the eth core is applied to a worker zone through their common host reference --
        // each maps its own ticks to host, then looks up.
        const int64_t b = base_ns(ticks);
        return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(b + detail::sync_correction_ns(chip_id_, b)));
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
    /** @brief Start and end of the zone on the host's std::chrono::steady_clock, converted through one d2d correction snapshot. */
    std::pair<std::chrono::steady_clock::time_point, std::chrono::steady_clock::time_point> host_span() const {
        const int64_t b0 = base_ns(timestamp_), b1 = base_ns(timestamp_ + duration_);
        int64_t d0 = 0, d1 = 0;
        detail::sync_correction_span_ns(chip_id_, b0, b1, d0, d1);
        return {
            std::chrono::steady_clock::time_point(std::chrono::nanoseconds(b0 + d0)),
            std::chrono::steady_clock::time_point(std::chrono::nanoseconds(b1 + d1))};
    }
};
static_assert(sizeof(Zone) == 48 && std::is_standard_layout_v<Zone>);

/**
 * @brief One DeviceTimestampedData marker with its values.
 *
 * Variable-length: to keep one past the callback, memcpy its size_bytes() bytes.
 */
class TimestampedData : public Record {
public:
    TimestampedData(const TimestampedData&) = delete;
    TimestampedData& operator=(const TimestampedData&) = delete;
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
[[nodiscard]] CallbackHandle RegisterCallback(std::string name, F callback) {
    using G = std::remove_reference_t<std::unwrap_reference_t<std::decay_t<F>>>;
    static_assert(detail::batch_callback<G>, "the callback must take one Batch<K> parameter");
    static_assert(std::is_copy_constructible_v<std::decay_t<F>>, "the callback must be copy-constructible");
    using A = detail::callback_param_t<G>;
    return detail::register_callback(
        std::move(name), [cb = std::move(callback)](const Batch<RecordType::All>& full) mutable {
            std::remove_cvref_t<A> batch(full);
            cb(std::forward<A>(batch));
        });
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
