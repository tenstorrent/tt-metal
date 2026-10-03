// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
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
//     Callback callback = RegisterCallback(
//         [](const Batch<RecordType::Zones>& b) {
//             for (const Zone& z : b.zones()) {
//                 use(z.site().name, z.core().logical, z.duration());
//             }
//         },
//         "my-tool");
//
namespace tt::tt_metal::experimental::streaming_profiler {

enum class Processor : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4, ERISC0 = 5, ERISC1 = 6 };

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
 * `logical` is the coordinate a program addresses it with; `physical` is its NoC 0 position on the die.
 * Tensix and Ethernet cores have separate logical grids, so two cores on a chip can share `logical` but never
 * `physical`.
 */
struct Core {
    CoreCoord logical;
    CoreCoord physical;
    ChipId chip_id = 0;
    Processor processor = Processor::BRISC;
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
class Zone;
class PointRecord;
class TimestampedData;
class Event;
class Callback;

// Implementation detail.
namespace detail {
constexpr bool has(RecordType set, RecordType type) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(type)) != 0;
}

inline constexpr uint32_t ZONE_ID_BITS = 27;
inline constexpr uint32_t ZONE_LOCAL_BITS = 14;
inline constexpr uint32_t ZONE_TU_COUNT = 1u << (ZONE_ID_BITS - ZONE_LOCAL_BITS);

struct SiteTu {
    std::span<const MarkerSite* const> sites;
};
extern std::atomic<const SiteTu*> site_tus[ZONE_TU_COUNT];
inline constexpr MarkerSite UNNAMED_SITE{.name = ""};

inline const MarkerSite& site_of(uint32_t zone_id) {
    const SiteTu* tu = site_tus[zone_id >> ZONE_LOCAL_BITS].load(std::memory_order_acquire);
    const uint32_t local = zone_id & ((1u << ZONE_LOCAL_BITS) - 1u);
    const MarkerSite* s = tu != nullptr && local < tu->sites.size() ? tu->sites[local] : nullptr;
    return s != nullptr ? *s : UNNAMED_SITE;
}

struct BatchData {
    const Zone* zones = nullptr;
    size_t zone_count = 0;
    const TimestampedData* timestamped_data = nullptr;
    size_t timestamped_data_count = 0;
    const Event* events = nullptr;
    size_t event_count = 0;
    uint64_t dropped_bytes = 0;
    uint64_t stall_count = 0;
};
enum class CallbackId : uint64_t {};
struct Access {
    template <RecordType K>
    static Batch<K> batch(const BatchData& data) {
        return Batch<K>(data);
    }
    static Callback callback(CallbackId id);
    static void construct_timestamped_data(
        void* at, const PointRecord& point, int64_t tsc, uint64_t value_count, const uint64_t* values);
};
CallbackId register_callback(std::string name, std::function<void(const BatchData&)> callback);

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

std::chrono::steady_clock::time_point tsc_to_steady(int64_t tsc);
}  // namespace detail

/** @brief Nanoseconds per host TSC tick. */
double NsPerTscTick() noexcept;

/** @brief Base class of every record: its site, core and program id. */
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
            .processor = processor_};
    }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const { return runtime_id_; }

protected:
    uint64_t timestamp_;
    uint32_t zone_id_;
    uint32_t runtime_id_;
    uint8_t logical_x_, logical_y_, physical_x_, physical_y_;
    uint16_t chip_id_;
    Processor processor_;
    int64_t tsc_;
};

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone with site name
 * STALL_ZONE_NAME.
 */
class Zone : public Record {
public:
    /** @brief When the device opened the zone, on steady_clock. */
    std::chrono::steady_clock::time_point start_time() const { return detail::tsc_to_steady(tsc_); }
    /** @brief When the device closed the zone, on steady_clock. */
    std::chrono::steady_clock::time_point end_time() const { return detail::tsc_to_steady(end_tsc_); }
    /** @brief When the device opened the zone, in host TSC ticks. */
    int64_t start_tsc() const { return tsc_; }
    /** @brief When the device closed the zone, in host TSC ticks. */
    int64_t end_tsc() const { return end_tsc_; }
    /** @brief When the zone opened, in device clock cycles. */
    uint64_t start_device_cycles() const { return timestamp_; }
    /** @brief When the zone closed, in device clock cycles. */
    uint64_t end_device_cycles() const { return timestamp_ + duration_; }
    /** @brief Length of the zone. */
    std::chrono::nanoseconds duration() const {
        return std::chrono::nanoseconds(std::llround(static_cast<double>(end_tsc_ - tsc_) * NsPerTscTick()));
    }
    /** @brief The device's average clock frequency over the zone. */
    double frequency_ghz() const {
        const int64_t tsc = end_tsc_ - tsc_;
        return tsc > 0 ? static_cast<double>(duration_) / (static_cast<double>(tsc) * NsPerTscTick()) : 0.0;
    }

private:
    uint64_t duration_;
    int64_t end_tsc_;
};

/** @brief Base class of records that mark one instant. */
class PointRecord : public Record {
public:
    /** @brief When the device recorded the marker, on steady_clock. */
    std::chrono::steady_clock::time_point time() const { return detail::tsc_to_steady(tsc_); }
    /** @brief When the device recorded the marker, in host TSC ticks. */
    int64_t tsc() const { return tsc_; }
    /** @brief When the device recorded the marker, in device clock cycles. */
    uint64_t device_cycles() const { return timestamp_; }
};

/** @brief One DeviceTimestampedData marker and the values it recorded. */
class TimestampedData : public PointRecord {
public:
    TimestampedData() = default;
    TimestampedData(const TimestampedData& other) : PointRecord(other), value_count_(other.value_count_) {
        uint64_t* values = new uint64_t[value_count_];
        std::ranges::copy(other.payload(), values);
        values_ = values;
    }
    TimestampedData(TimestampedData&& other) noexcept :
        PointRecord(other),
        value_count_(std::exchange(other.value_count_, 0)),
        values_(std::exchange(other.values_, nullptr)) {}
    TimestampedData& operator=(TimestampedData other) noexcept {
        PointRecord::operator=(other);
        std::swap(value_count_, other.value_count_);
        std::swap(values_, other.values_);
        return *this;
    }
    ~TimestampedData() { delete[] values_; }

    /** @brief The marker's values. */
    std::span<const uint64_t> payload() const { return {values_, value_count_}; }

private:
    friend struct detail::Access;
    TimestampedData(const PointRecord& record, int64_t tsc, uint64_t value_count, const uint64_t* values) :
        PointRecord(record), value_count_(value_count), values_(values) {
        tsc_ = tsc;
    }

    uint64_t value_count_ = 0;
    const uint64_t* values_ = nullptr;
};

inline void detail::Access::construct_timestamped_data(
    void* at, const PointRecord& point, int64_t tsc, uint64_t value_count, const uint64_t* values) {
    new (at) TimestampedData(point, tsc, value_count, values);
}

/** @brief One DeviceRecordEvent marker. */
class Event : public PointRecord {};

/**
 * @brief The records of each type in K.
 *
 * Records from one processor are in the order it emitted them, so its zones are ordered by end time. The records are
 * valid only inside the callback; to keep one, copy it.
 */
template <RecordType K>
class Batch {
public:
    std::span<const Zone> zones() const
        requires(detail::has(K, RecordType::Zones))
    {
        return {data_.zones, data_.zone_count};
    }
    std::span<const TimestampedData> timestamped_data() const
        requires(detail::has(K, RecordType::TimestampedData))
    {
        return {data_.timestamped_data, data_.timestamped_data_count};
    }
    std::span<const Event> events() const
        requires(detail::has(K, RecordType::Events))
    {
        return {data_.events, data_.event_count};
    }
    /**
     * @brief Bytes of device output lost since this callback last ran; nonzero if the callback could not keep up with
     *        incoming data.
     *
     * Raising TT_METAL_STREAMING_PROFILER_FIFO_MB lets a callback fall further behind before it loses data.
     */
    uint64_t dropped_bytes() const { return data_.dropped_bytes; }
    /**
     * @brief Stalls on any core since the previous batch: times a core waited for the profiler to drain its records.
     */
    uint64_t stall_count() const { return data_.stall_count; }

private:
    friend struct detail::Access;
    explicit Batch(const detail::BatchData& data) : data_(data) {}

    detail::BatchData data_;
};

/** @brief A copy-constructible callable taking one `const Batch<K>&`. */
template <typename F>
concept BatchCallable = detail::accepted_batch<F>() != RecordType{} && std::is_copy_constructible_v<F>;

/**
 * @brief A registered callback, which stays registered until this object is destroyed or assigned over.
 *
 * If the callback is running, unregistering waits for it to return. A callback may unregister only itself, which
 * returns at once and delivers it no further batches.
 */
class [[nodiscard]] Callback {
public:
    Callback() = default;
    Callback(Callback&& other) noexcept : id_(std::exchange(other.id_, detail::CallbackId{})) {}
    Callback& operator=(Callback&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = std::exchange(other.id_, detail::CallbackId{});
        }
        return *this;
    }
    ~Callback() { reset(); }

private:
    friend struct detail::Access;
    explicit Callback(detail::CallbackId id) : id_(id) {}
    void reset() noexcept;
    detail::CallbackId id_{};
};

inline Callback detail::Access::callback(CallbackId id) { return Callback(id); }

/**
 * @brief Registers a callback to be invoked when streaming profiler records arrive from a device.
 *
 * Multiple callbacks can be registered; each runs on its own thread, one invocation at a time. If a callback shares a
 * resource with other callbacks, access it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep
 * up with incoming data miss records; this is reported by Batch::dropped_bytes. May be called before, during or
 * between captures.
 *
 * @param callback Takes one `const Batch<K>&`; K selects the record types it receives.
 * @param name Optional name for the callback in the profiler's logs and thread names.
 * @return The registration; the callback runs until it is destroyed.
 */
template <BatchCallable F>
Callback RegisterCallback(F callback, std::string name = {}) {
    constexpr RecordType kAccepted = detail::accepted_batch<F>();
    return detail::Access::callback(detail::register_callback(
        std::move(name), [callback = std::move(callback)](const detail::BatchData& data) mutable {
            callback(detail::Access::batch<kAccepted>(data));
        }));
}

/** @brief Returns true if the streaming profiler is currently running on at least one chip. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
