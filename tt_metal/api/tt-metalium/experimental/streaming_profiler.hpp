// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt_stl/strong_type.hpp>

// The streaming profiler's host API. A callback registered here receives the profiler records that device kernels emit.
//
// This API is experimental and may change or be removed without notice.
//
//     Callback callback = RegisterCallback(
//         [](const Batch<Zone>& batch) {
//             for (const Zone& zone : batch.records<Zone>()) {
//                 use(zone.site().name, zone.core().logical, zone.duration());
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
 * Tensix and Ethernet cores have separate logical grids, so two cores on a device can share `logical` but never
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

class Zone;
class TimestampedData;
class Event;
class Callback;

// Implementation detail.
namespace detail {
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

struct BatchData;
using CallbackId = ttsl::StrongType<uint32_t, struct CallbackIdTag>;
Callback register_callback(std::string name, uint32_t types, std::function<void(const BatchData&)> callback);
using RecordSpans = std::tuple<std::span<const Zone>, std::span<const TimestampedData>, std::span<const Event>>;
template <typename T>
inline constexpr uint32_t record_bit = []<std::size_t... I>(std::index_sequence<I...>) {
    return ((std::same_as<std::tuple_element_t<I, RecordSpans>, std::span<const T>> ? 1u << I : 0u) | ...);
}(std::make_index_sequence<std::tuple_size_v<RecordSpans>>());
}  // namespace detail

/** @brief Base class of every record, holding its site, core and program runtime ID. */
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
    static std::chrono::steady_clock::time_point tsc_to_steady(int64_t tsc) {
        const TscToSteadyLine& line = tsc_to_steady_line_;
        if (tsc < line.from || tsc > line.to) {
            return refill_tsc_to_steady_line(tsc);
        }
        return line.at(tsc);
    }

    uint64_t device_cycles_ = 0;
    uint32_t zone_id_ = 0;
    uint32_t runtime_id_ = 0;
    uint8_t logical_x_ = 0, logical_y_ = 0, physical_x_ = 0, physical_y_ = 0;
    uint16_t chip_id_ = 0;
    Processor processor_ = Processor::BRISC;
    int64_t tsc_ = 0;

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
};

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone with site name
 * STALL_ZONE_NAME.
 */
class Zone : public Record {
public:
    /** @brief When the zone opened, on steady_clock. */
    std::chrono::steady_clock::time_point start_time() const { return tsc_to_steady(tsc_); }
    /** @brief When the zone closed, on steady_clock. */
    std::chrono::steady_clock::time_point end_time() const { return tsc_to_steady(end_tsc_); }
    /** @brief When the zone opened, in host reference cycles (the x86 time-stamp counter). */
    int64_t start_host_cycles() const { return tsc_; }
    /** @brief When the zone closed, in host reference cycles (the x86 time-stamp counter). */
    int64_t end_host_cycles() const { return end_tsc_; }
    /** @brief When the zone opened, in the core's own clock cycles. */
    uint64_t start_device_cycles() const { return device_cycles_; }
    /** @brief When the zone closed, in the core's own clock cycles. */
    uint64_t end_device_cycles() const { return device_cycles_ + duration_cycles_; }
    /** @brief Length of the zone. */
    std::chrono::nanoseconds duration() const { return end_time() - start_time(); }
    /** @brief The device's average clock frequency over the zone. */
    double frequency_ghz() const {
        return static_cast<double>(duration_cycles_) / static_cast<double>(duration().count());
    }

private:
    uint64_t duration_cycles_ = 0;
    int64_t end_tsc_ = 0;
};

/** @brief Base class of records that mark one instant. */
class PointRecord : public Record {
public:
    /** @brief When the marker was recorded, on steady_clock. */
    std::chrono::steady_clock::time_point time() const { return tsc_to_steady(tsc_); }
    /** @brief When the marker was recorded, in host reference cycles (the x86 time-stamp counter). */
    int64_t host_cycles() const { return tsc_; }
    /** @brief When the marker was recorded, in the core's own clock cycles. */
    uint64_t device_cycles() const { return device_cycles_; }
};

/** @brief One DeviceTimestampedData marker and the values it recorded. */
class TimestampedData : public PointRecord {
public:
    TimestampedData() = default;
    TimestampedData(const TimestampedData& other);
    TimestampedData(TimestampedData&& other) noexcept;
    TimestampedData& operator=(TimestampedData other) noexcept;
    ~TimestampedData();

    /** @brief The marker's values. */
    std::span<const uint64_t> payload() const { return {values_, value_count_}; }

private:
    uint64_t value_count_ = 0;
    const uint64_t* values_ = nullptr;
};

/** @brief One DeviceRecordEvent marker. */
class Event : public PointRecord {};

/** @brief A type a Batch can hold: Zone, TimestampedData or Event. */
template <typename T>
concept RecordType = detail::record_bit<T> != 0;

template <RecordType... Ts>
    requires(sizeof...(Ts) > 0)
class Batch;

// Implementation detail.
namespace detail {
struct BatchData {
    RecordSpans records;
    uint64_t dropped_bytes = 0;
    uint64_t stall_count = 0;
};
template <typename Signature>
struct BatchParameter {};
template <typename R, typename... Ts>
struct BatchParameter<std::function<R(const Batch<Ts...>&)>> {
    using type = Batch<Ts...>;
};
template <typename R, typename... Ts>
struct BatchParameter<std::function<R(Batch<Ts...>)>> {
    using type = Batch<Ts...>;
};
template <typename F>
using CallbackBatch =
    typename BatchParameter<decltype(std::function{std::declval<std::unwrap_ref_decay_t<F>>()})>::type;
}  // namespace detail

/** @brief A copy-constructible callable taking one `const Batch<Ts...>&`. */
template <typename F>
concept BatchCallable = requires { typename detail::CallbackBatch<F>; } && std::is_copy_constructible_v<F>;

/**
 * @brief A registered callback, which stays registered until this object is destroyed, reset or assigned over.
 *
 * If the callback is running, unregistering waits for it to return.
 */
class [[nodiscard]] Callback {
public:
    Callback() = default;
    Callback(Callback&& other) noexcept;
    Callback& operator=(Callback&& other) noexcept;
    ~Callback();

    /** @brief Unregisters the callback. */
    void reset() noexcept;

private:
    friend Callback detail::register_callback(
        std::string name, uint32_t types, std::function<void(const detail::BatchData&)> callback);
    explicit Callback(detail::CallbackId id) : id_(id) {}
    detail::CallbackId id_;
};

/**
 * @brief Registers a callback to be invoked when streaming profiler records arrive from a device.
 *
 * Multiple callbacks can be registered; each runs on its own thread, one invocation at a time. If a callback shares a
 * resource with other callbacks, access it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep
 * up with incoming data miss records; this is reported by Batch::dropped_bytes. TT_METAL_STREAMING_PROFILER_FIFO_MB
 * sets how far a callback can fall behind before it misses records. May be called before, during or between captures.
 *
 * @param callback Takes one `const Batch<Ts...>&`; Ts selects the record types it receives.
 * @param name Optional name for the callback in the profiler's logs and thread names.
 * @return The registration; the callback runs until it is destroyed.
 */
template <BatchCallable F>
Callback RegisterCallback(F callback, std::string name = {}) {
    return detail::register_callback(
        std::move(name),
        detail::CallbackBatch<F>::kTypes,
        [callback = std::move(callback)](const detail::BatchData& data) mutable {
            callback(detail::CallbackBatch<F>(data));
        });
}

/**
 * @brief A batch of records passed to a callback.
 *
 * Records from one processor are in the order it emitted them (for zones, the order they closed). The records are valid
 * only until the callback returns; to keep one, copy it.
 */
template <RecordType... Ts>
    requires(sizeof...(Ts) > 0)
class Batch {
public:
    /** @brief The batch's records of type T. */
    template <typename T>
        requires(std::same_as<T, Ts> || ...)
    std::span<const T> records() const {
        return std::get<std::span<const T>>(data_.records);
    }
    /**
     * @brief Bytes of device output lost since the callback last ran.
     *
     * Nonzero if the callback could not keep up with incoming data. Raising TT_METAL_STREAMING_PROFILER_FIFO_MB lets a
     * callback fall further behind before it loses data.
     */
    uint64_t dropped_bytes() const { return data_.dropped_bytes; }
    /** @brief How many times the cores in this batch waited for the profiler to drain their records. */
    uint64_t stall_count() const { return data_.stall_count; }

private:
    template <BatchCallable F>
    friend Callback RegisterCallback(F callback, std::string name);
    static constexpr uint32_t kTypes = (detail::record_bit<Ts> | ...);
    explicit Batch(const detail::BatchData& data) : data_(data) {}

    detail::BatchData data_;
};

/** @brief Returns true if the streaming profiler is currently running on at least one device. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
