// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/vocabulary/device_types.hpp>

// The streaming profiler's host API. A callback registered here receives the profiler records that device kernels emit.
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

enum class Processor : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4, ERISC0 = 5, ERISC1 = 6 };

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


}

}  // namespace detail

/** @brief Base class of every record, holding its observability event's descriptor id, core and program runtime ID. */
class Record {
public:
    /** @brief The descriptor of the observability event this record came from. */
    /** @brief The core that emitted the record. */
    }
    /** @brief Host runtime ID of the program. */

protected:
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
    /** @brief When the zone opened, on steady_clock. */
    /** @brief When the zone closed, on steady_clock. */
    /** @brief When the zone opened, in host reference cycles (the x86 time-stamp counter). */
    /** @brief When the zone closed, in host reference cycles (the x86 time-stamp counter). */
    /** @brief When the zone opened, in the core's own clock cycles. */
    /** @brief When the zone closed, in the core's own clock cycles. */
    /** @brief Length of the zone. */
    /** @brief The device's average clock frequency over the zone. */
    double frequency_ghz() const {
    }

private:
};

/** @brief Base class of records that mark one instant. */
class PointRecord : public Record {
public:
    /** @brief When the event was recorded, on steady_clock. */
    /** @brief When the event was recorded, in host reference cycles (the x86 time-stamp counter). */
    /** @brief When the event was recorded, in the core's own clock cycles. */
};

/** @brief One DeviceTimestampedData event and the values it recorded. */
class TimestampedData : public PointRecord {
public:
    TimestampedData() = default;
    /** @brief The event's values. */

private:
};

/** @brief One DeviceRecordEvent. */

/** @brief A type a Batch can hold: Zone, TimestampedData or Event. */
template <typename T>

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
 */

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
    }
    /**
     * @brief Bytes of device output lost since the callback last ran.
     *
     * Nonzero if the callback could not keep up with incoming data. Raising TT_METAL_STREAMING_PROFILER_FIFO_MB lets a
     * callback fall further behind before it loses data.
     */
    /** @brief How many times the cores in this batch waited for the profiler to drain their records. */

private:
    template <BatchCallable F>
    friend Callback RegisterCallback(F callback, std::string name);

};

/** @brief Returns true if the streaming profiler is currently running on at least one device. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
