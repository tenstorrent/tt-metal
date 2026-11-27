// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/vocabulary/device_types.hpp>
#include <tt/insight/tools/profiler/host/records.hpp>

// The streaming profiler's host API. A callback registered here receives the profiler records that device kernels emit.
// Each record is a view of the tt-insight record it stands for, with the core Metal knows its publisher by: in a batch
// it points at tt-insight's record, and a copy of it owns one.
//
// This API is experimental and may change or be removed without notice.
//
//     Callback callback = RegisterCallback(
//         [](const Batch<Zone>& batch) {
//             for (const Zone& zone : batch.records<Zone>()) {
//                 use(zone.descriptor().name, zone.core().logical, zone.duration());
//             }
//         },
//         "my-tool");
//
namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief A processor of a Tensix or Ethernet core. */
enum class Processor : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4, ERISC0 = 5, ERISC1 = 6 };

/** @brief A location in device-side source code. */
struct SourceLocation {
    std::string_view file;  // Valid for the lifetime of the process.
    uint32_t line = 0;
};

/** @brief An observability event's name and where it is declared in device-side source code. */
struct ObservabilityEventDescriptor {
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

/** @brief Name of a stall zone. */
inline constexpr std::string_view STALL_ZONE_NAME = "PROFILER-STALL";

// Implementation detail.
namespace detail {
template <template <typename...> typename T>
using Insight =
    T<tt::insight::profiler::Zone,
      tt::insight::Stall,
      tt::insight::profiler::Event,
      tt::insight::profiler::TimestampedData>;
using InsightBatch = Insight<tt::insight::Batch>;
using InsightRecord = Insight<std::variant>;
// A tt-insight record type's index among them.
template <typename R>
inline constexpr uintptr_t kind_of = std::same_as<R, tt::insight::Stall>                       ? 1
                                     : std::same_as<R, tt::insight::profiler::Event>           ? 2
                                     : std::same_as<R, tt::insight::profiler::TimestampedData> ? 3
                                                                                               : 0;

// Metal's core at each tile of each chip, [chip][y * 64 + x] in translated coordinates, set before the chip's first
// capture starts and kept for the process: a callback reads it only for records of a capture started after.
extern std::array<const std::array<Core, 64 * 64>*, 64> chip_cores;

// Metal's name for a processor of tt-insight's. Metal's programs publish from these seven alone.
inline Processor processor_of(tt::insight::Processor::Type type) {
    switch (type) {
        case tt::insight::Processor::NCRISC: return Processor::NCRISC;
        case tt::insight::Processor::TRISC0: return Processor::TRISC0;
        case tt::insight::Processor::TRISC1: return Processor::TRISC1;
        case tt::insight::Processor::TRISC2: return Processor::TRISC2;
        case tt::insight::Processor::ERISC0: return Processor::ERISC0;
        case tt::insight::Processor::ERISC1: return Processor::ERISC1;
        default: return Processor::BRISC;
    }
}

inline Core core_of(const tt::insight::Processor& publisher) {
    const auto* cores = publisher.device_id < chip_cores.size() ? chip_cores[publisher.device_id] : nullptr;
    Core core = cores != nullptr ? (*cores)[(publisher.tile.y * 64) + publisher.tile.x]
                                 : Core{.chip_id = static_cast<ChipId>(publisher.device_id)};
    core.processor = processor_of(publisher.type);
    return core;
}

// Where a record was published, which tt-insight looks up. Pure, out of line and returning a pointer, so that the
// compiler drops the lookup for a caller that reads only a descriptor's name.
template <typename R>
[[gnu::pure, gnu::noinline]] const tt::insight::SourceLocation* location_of(const R& record) noexcept {
    return &record.source_location();
}
// A stall has no source location.
inline const tt::insight::SourceLocation* location_of(const tt::insight::Stall&) noexcept {
    static constexpr tt::insight::SourceLocation kNone{};
    return &kNone;
}
}  // namespace detail

/** @brief Base class of every record, holding its observability event's descriptor id, core and program runtime ID. */
class Record {
protected:
    // tt-insight's record: its address, with its kind in the two low bits and kOwned set in a copy, which owns the
    // record it points at. One word, so that a batch's records are written and read as an array of addresses.
    static constexpr uintptr_t kKind = 3, kOwned = 4;
    uintptr_t record_ = 0;
    template <size_t K>
    const std::variant_alternative_t<K, detail::InsightRecord>& as() const {
        return *reinterpret_cast<const std::variant_alternative_t<K, detail::InsightRecord>*>(record_ & ~uintptr_t{7});
    }
    // Calls `f` with tt-insight's record, always inlined, so that a caller pays only for the fields it reads. Two
    // two-way choices, which the compiler folds into one where the kinds read the same field the same way.
    template <typename F>
    [[gnu::always_inline]] decltype(auto) visit(F&& f) const {
        if ((record_ & 2) == 0) {
            return (record_ & 1) == 0 ? f(as<0>()) : f(as<1>());
        }
        return (record_ & 1) == 0 ? f(as<2>()) : f(as<3>());
    }

public:
    /** @brief The descriptor of the observability event this record came from. */
    [[gnu::always_inline]] ObservabilityEventDescriptor descriptor() const {
        return visit([](const auto& r) __attribute__((always_inline)) {
            const tt::insight::SourceLocation* location = detail::location_of(r);
            return ObservabilityEventDescriptor{name(r), {location->file, location->line}};
        });
    }
    /** @brief The core that emitted the record. */
    [[gnu::always_inline]] Core core() const {
        return visit([](const auto& r) __attribute__((always_inline)) { return detail::core_of(r.publisher()); });
    }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const {
        return visit([](const auto& r) { return r.context_id(); });
    }

    // A copy owns its tt-insight record, so it can be kept after the callback returns.
    Record(const Record& other) :
        record_(other.visit([&](const auto& r) {
            return reinterpret_cast<uintptr_t>(new std::remove_cvref_t<decltype(r)>(r)) | (other.record_ & kKind) |
                   kOwned;
        })) {}
    Record(Record&& other) noexcept : record_(std::exchange(other.record_, other.record_ & ~kOwned)) {}
    Record& operator=(Record other) noexcept {
        std::swap(record_, other.record_);
        return *this;
    }
    ~Record() {
        if ((record_ & kOwned) != 0) {
            visit([](const auto& r) { delete &r; });
        }
    }

protected:
    // A record of a batch: tt-insight's record lives until the callback returns.
    Record() = default;
    template <typename R>
        requires(!std::derived_from<R, Record>)
    explicit Record(const R& record) : record_(reinterpret_cast<uintptr_t>(&record) | detail::kind_of<R>) {
        static_assert(
            alignof(R) >= 8 && std::same_as<R, std::variant_alternative_t<detail::kind_of<R>, detail::InsightRecord>>);
    }
    static std::string_view name(const tt::insight::Stall&) { return STALL_ZONE_NAME; }
    static std::string_view name(const auto& record) { return record.metadata().name; }
    const tt::insight::Timestamp& published_at() const {
        return visit([](const auto& r) -> const tt::insight::Timestamp& { return r.published_at(); });
    }
};

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone with name
 * STALL_ZONE_NAME.
 */
class Zone : public Record {
public:
    /** @brief An empty record, to be assigned one before it is read. */
    Zone() = default;
    explicit Zone(const tt::insight::profiler::Zone& zone) : Record(zone) {}
    explicit Zone(const tt::insight::Stall& stall) : Record(stall) {}
    /** @brief When the zone opened, on steady_clock. */
    std::chrono::steady_clock::time_point start_time() const { return interval().start.time(); }
    /** @brief When the zone closed, on steady_clock. */
    std::chrono::steady_clock::time_point end_time() const { return interval().end.time(); }
    /** @brief When the zone opened, in host reference cycles (the x86 time-stamp counter). */
    int64_t start_host_cycles() const { return static_cast<int64_t>(interval().start.host_tsc()); }
    /** @brief When the zone closed, in host reference cycles (the x86 time-stamp counter). */
    int64_t end_host_cycles() const { return static_cast<int64_t>(interval().end.host_tsc()); }
    /** @brief When the zone opened, in the core's own clock cycles. */
    uint64_t start_device_cycles() const { return interval().start.device_cycles(); }
    /** @brief When the zone closed, in the core's own clock cycles. */
    uint64_t end_device_cycles() const { return interval().end.device_cycles(); }
    /** @brief Length of the zone. */
    std::chrono::nanoseconds duration() const { return interval().duration(); }
    /** @brief The device's average clock frequency over the zone. */
    double frequency_ghz() const {
        return static_cast<double>(end_device_cycles() - start_device_cycles()) /
               static_cast<double>(duration().count());
    }

private:
    const tt::insight::Interval& interval() const {
        return (record_ & kKind) == 0 ? as<0>().payload() : as<1>().payload();
    }
};

/** @brief Base class of records that mark one instant. */
class PointRecord : public Record {
public:
    /** @brief When the event was recorded, on steady_clock. */
    std::chrono::steady_clock::time_point time() const { return published_at().time(); }
    /** @brief When the event was recorded, in host reference cycles (the x86 time-stamp counter). */
    int64_t host_cycles() const { return static_cast<int64_t>(published_at().host_tsc()); }
    /** @brief When the event was recorded, in the core's own clock cycles. */
    uint64_t device_cycles() const { return published_at().device_cycles(); }

protected:
    PointRecord() = default;
    using Record::Record;
};

/** @brief One DeviceTimestampedData event and the values it recorded. */
class TimestampedData : public PointRecord {
public:
    TimestampedData() = default;
    explicit TimestampedData(const tt::insight::profiler::TimestampedData& data) : PointRecord(data) {
        for (const tt::insight::profiler::Number& number : data.payload()) {
            values_.push_back(std::visit([](auto value) { return static_cast<uint64_t>(value); }, number));
        }
    }
    /** @brief The event's values. */
    std::span<const uint64_t> payload() const { return values_; }

private:
    std::vector<uint64_t> values_;
};

/** @brief One DeviceRecordEvent. */
class Event : public PointRecord {
public:
    Event() = default;
    explicit Event(const tt::insight::profiler::Event& event) : PointRecord(event) {}
};

/** @brief A type a Batch can hold: Zone, TimestampedData or Event. */
template <typename T>
concept RecordType = std::same_as<T, Zone> || std::same_as<T, TimestampedData> || std::same_as<T, Event>;

template <RecordType... Ts>
    requires(sizeof...(Ts) > 0)
class Batch;

// Implementation detail.
namespace detail {
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
 * Unregistering waits for a call in progress to return. A callback has received every record published before it once
 * it is unregistered: to read what it gathered, close the devices, unregister it, and read then.
 */
using Callback = tt::insight::Callback;

/**
 * @brief Registers a callback to be invoked when streaming profiler records arrive from a device.
 *
 * Multiple callbacks can be registered; each runs on its own thread, one invocation at a time. If a callback shares a
 * resource with other callbacks, access it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep
 * up with incoming data miss records; this is reported by Batch::dropped_bytes. TT_METAL_STREAMING_PROFILER_FIFO_MB,
 * the host memory that holds each device's records, sets how far a callback can fall behind before it misses records.
 * May be called before, during or between captures.
 *
 * @param callback Takes one `const Batch<Ts...>&`; Ts selects the record types it receives.
 * @param name Optional name for the callback in the profiler's logs and thread names.
 * @return The registration; the callback runs until it is destroyed.
 */
template <BatchCallable F>
Callback RegisterCallback(F callback, std::string name = {}) {
    return tt::insight::register_callback(
        [callback = std::move(callback),
         batch = detail::CallbackBatch<F>()](const detail::InsightBatch& records) mutable {
            batch.assign(records);
            callback(std::as_const(batch));
        },
        std::move(name));
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
        return {std::get<std::vector<T>>(records_).data(), std::get<T*>(ends_)};
    }
    /**
     * @brief Bytes of device output lost since the callback last ran.
     *
     * Nonzero if the callback could not keep up with incoming data. Raising TT_METAL_STREAMING_PROFILER_FIFO_MB, the
     * host memory that holds each device's records, lets a callback fall further behind before it loses data.
     */
    uint64_t dropped_bytes() const { return batch_->lost_byte_count(); }
    /** @brief How many times the cores in this batch waited for the profiler to drain their records. */
    uint64_t stall_count() const { return stall_count_; }

private:
    template <BatchCallable F>
    friend Callback RegisterCallback(F callback, std::string name);
    Batch() = default;

    template <typename T, typename... Rs>
    static size_t room(const tt::insight::Batch<Rs...>& batch) {
        return ((std::is_constructible_v<T, const Rs&> ? batch.template records<Rs>().size() : 0) + ...);
    }

    // A vector keeps its elements from batch to batch and only grows, to the most records its type can take of a
    // batch (room). A batch's records overwrite them in tt-insight's order, and `ends_` says where each type's end.
    void assign(const detail::InsightBatch& batch) {
        batch_ = &batch;
        stall_count_ = batch.records<tt::insight::Stall>().size();
        const auto first = [&]<typename T>(std::vector<T>& items) {
            items.resize(std::max(items.size(), room<T>(batch)));
            return items.data();
        };
        std::tuple<Ts*...> next{first(std::get<std::vector<Ts>>(records_))...};
        batch.visit([&next]<typename R>(const R& record) {
            (
                [&] {
                    if constexpr (std::same_as<Ts, TimestampedData> && std::is_constructible_v<Ts, const R&>) {
                        *std::get<Ts*>(next)++ = Ts(record);
                    } else if constexpr (std::is_constructible_v<Ts, const R&>) {
                        // The element is an earlier batch's, which owns nothing: it is written over as it is.
                        std::construct_at(std::get<Ts*>(next)++, record);
                    }
                }(),
                ...);
        });
        ends_ = next;
    }

    const detail::InsightBatch* batch_ = nullptr;
    uint64_t stall_count_ = 0;
    std::tuple<std::vector<Ts>...> records_;
    std::tuple<Ts*...> ends_;
};

/** @brief Returns true if the streaming profiler is currently running on at least one device. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
