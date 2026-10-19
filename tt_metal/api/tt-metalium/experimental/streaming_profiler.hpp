// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <memory_resource>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt_stl/strong_type.hpp>
#include <hostdevcommon/streaming_profiler_reflection.hpp>
#include <hostdevcommon/streaming_profiler_types.hpp>

// The streaming profiler's host API. A callback registered here receives the profiler records that device kernels emit.
//
// This API is experimental and may change or be removed without notice.
//
//     Callback callback = RegisterCallback(
//         [](const Batch<Zone>& batch) {
//             for (const Zone& zone : batch.records<Zone>()) {
//                 use(zone.metadata().name, zone.core().logical, zone.payload().duration());
//             }
//         },
//         "my-tool");
//
namespace tt::tt_metal::experimental::streaming_profiler {

enum class Processor : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4, ERISC0 = 5, ERISC1 = 6 };

/**
 * @brief A location in kernel source code.
 *
 * The strings are valid for the life of the process.
 */
struct SourceLocation {
    std::string_view file;
    std::string_view function;
    uint32_t line = 0;
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

/**
 * @brief The payload of a record sent as one or more arguments, `DevicePublish<M>(a, b, ...)`: a list of T, which is
 * the arguments' type or a std::variant of the types they may have. Among those types, a std::span<const E> takes an
 * array argument of any length and a std::array<E, N> one of exactly N elements.
 */
template <typename T>
struct Arguments {
    using value_type = T;
};

class Callback;

namespace detail {
struct AnyMetadata {};
template <typename T>
inline constexpr bool is_record = false;
}  // namespace detail

template <typename Payload, typename Metadata = detail::AnyMetadata>
class Record;

template <typename Payload, typename Metadata>
inline constexpr bool detail::is_record<Record<Payload, Metadata>> = true;

/** @brief A type a Batch can hold: a Record. */
template <typename T>
concept RecordType = detail::is_record<T>;

template <RecordType... Ts>
    requires(sizeof...(Ts) > 0)
class Batch;

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone named
 * STALL_ZONE_NAME.
 */
using Zone = Record<Interval, ZoneMetadata>;
/** @brief One DeviceTimestampedData marker and the values it recorded. */
using TimestampedData = Record<Arguments<uint64_t>, TimestampedDataMetadata>;
/** @brief One DeviceRecordEvent marker. */
using Event = Record<void, EventMetadata>;

// Implementation detail.
namespace detail {
const SourceLocation& location_of(uint32_t zone_id);

struct BatchData {
    uint64_t dropped_bytes = 0;
    uint64_t stall_count = 0;
    std::span<const std::byte* const> records;
    std::span<const std::size_t> counts;
    std::span<const uint32_t> order;
};
struct Kind;
using CallbackId = ttsl::StrongType<uint32_t, struct CallbackIdTag>;
Callback register_callback(
    std::string name, std::span<const Kind> kinds, std::function<void(const BatchData&)> callback);

class FieldReader {
public:
    virtual ~FieldReader() = default;
    void open() {}
    void close() {}
    template <typename T>
    T leaf() {
        T value{};
        read(&value);
        return value;
    }

    // A record's values: each one's alternative and field count, and the arena their arrays go in.
    const std::span<const uint8_t> alternatives;
    const std::span<const uint32_t> value_fields;
    std::pmr::memory_resource* const arena;

protected:
    explicit FieldReader(
        std::span<const uint8_t> alternatives = {},
        std::span<const uint32_t> value_fields = {},
        std::pmr::memory_resource* arena = nullptr) :
        alternatives(alternatives), value_fields(value_fields), arena(arena) {}
    virtual void read(void* out) = 0;
};

template <typename T>
inline constexpr bool is_array_value = false;
template <typename E>
inline constexpr bool is_array_value<std::span<const E>> = true;
template <typename T>
inline constexpr bool is_variant = false;
template <typename... Ts>
inline constexpr bool is_variant<std::variant<Ts...>> = true;

struct Source {
    uint32_t zone_id = 0;
    uint32_t runtime_id = 0;
    uint8_t logical_x = 0, logical_y = 0, physical_x = 0, physical_y = 0;
    uint16_t chip_id = 0;
    Processor processor = Processor::BRISC;
};
// An Arguments<V> record's arguments, built in the batch's arena.
template <typename V>
class RecordArguments {
public:
    RecordArguments() = default;
    explicit RecordArguments(FieldReader& data) {
        V* values = std::pmr::polymorphic_allocator<V>(data.arena).allocate(data.alternatives.size());
        for (std::size_t k = 0; k < data.alternatives.size(); k++) {
            std::construct_at(values + k, build_value(data, data.alternatives[k], data.value_fields[k]));
        }
        values_ = {values, data.alternatives.size()};
    }
    // A copy owns its values and their arrays, so it outlives the batch.
    RecordArguments(const RecordArguments& other) : owned_(other.values_.begin(), other.values_.end()) {
        for (V& value : owned_) {
            if constexpr (is_variant<V>) {
                std::visit([&](auto& alternative) { own_array(alternative); }, value);
            } else {
                own_array(value);
            }
        }
        values_ = owned_;
    }
    RecordArguments(RecordArguments&&) noexcept = default;
    RecordArguments& operator=(const RecordArguments& other) { return *this = RecordArguments(other); }
    RecordArguments& operator=(RecordArguments&&) noexcept = default;

    const std::span<const V>& get() const { return values_; }

private:
    template <typename A>
    static A build_alternative(FieldReader& data, std::size_t fields) {
        if constexpr (is_array_value<A>) {
            using E = typename A::value_type;
            const std::size_t count = fields / leaf_codes<E>.size();
            E* elements = std::pmr::polymorphic_allocator<E>(data.arena).allocate(count);
            for (std::size_t k = 0; k < count; k++) {
                std::construct_at(elements + k, build<E>(data));
            }
            return A(elements, count);
        } else {
            return build<A>(data);
        }
    }
    static V build_value(FieldReader& data, std::size_t alternative, std::size_t fields) {
        if constexpr (is_variant<V>) {
            V value;
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                static_cast<void>(
                    ((alternative == I &&
                      (value.template emplace<I>(build_alternative<std::variant_alternative_t<I, V>>(data, fields)),
                       true)) ||
                     ...));
            }(std::make_index_sequence<std::variant_size_v<V>>());
            return value;
        } else {
            return build_alternative<V>(data, fields);
        }
    }
    template <typename A>
    void own_array(A& value) {
        if constexpr (is_array_value<A>) {
            using E = typename A::value_type;
            auto elements = std::make_shared<E[]>(value.size());
            std::ranges::copy(value, elements.get());
            value = A(elements.get(), value.size());
            arrays_.push_back(std::move(elements));
        }
    }

    std::vector<V> owned_;
    std::span<const V> values_;
    std::vector<std::shared_ptr<const void>> arrays_;
};
template <typename T>
struct Stored {
    using type = T;
};
template <>
struct Stored<void> {
    using type = std::monostate;
};
template <typename V>
struct Stored<Arguments<V>> {
    using type = RecordArguments<V>;
};
template <typename T>
inline constexpr bool is_arguments = false;
template <typename V>
inline constexpr bool is_arguments<Arguments<V>> = true;

struct ValueType {
    std::string_view name;
    std::span<const uint16_t> codes;
    bool array = false;
    std::size_t count = 0;  // an array's elements, or 0 for any number
};
template <typename T>
inline constexpr ValueType value_type{
    is_leaf<T> && !std::is_enum_v<T> ? std::string_view() : type_name<T>, leaf_codes<T>};
template <typename E>
inline constexpr ValueType value_type<std::span<const E>>{value_type<E>.name, leaf_codes<E>, true};
template <typename E, std::size_t N>
inline constexpr ValueType value_type<std::array<E, N>>{value_type<E>.name, leaf_codes<E>, true, N};
template <typename T>
inline constexpr std::array<ValueType, 0> value_types{};
template <typename V>
inline constexpr std::array<ValueType, 1> value_types<Arguments<V>>{value_type<V>};
template <typename... Ts>
inline constexpr std::array<ValueType, sizeof...(Ts)> value_types<Arguments<std::variant<Ts...>>>{value_type<Ts>...};

// A callback's data and metadata types follow the rules kernels check, so a type no kernel could send fails here.
template <typename V>
struct DeviceValue {
    using type = V;
};
template <typename E>
struct DeviceValue<std::span<const E>> {
    using type = std::array<typename DeviceValue<E>::type, 1>;
};
template <typename V>
inline constexpr bool values_follow_rules = check_value<typename DeviceValue<V>::type>();
template <typename... Vs>
inline constexpr bool values_follow_rules<std::variant<Vs...>> =
    (check_value<typename DeviceValue<Vs>::type>() & ... & true);
template <typename T>
constexpr bool follows_rules() {
    if constexpr (std::is_void_v<T>) {
        return true;
    } else if constexpr (is_arguments<T>) {
        return values_follow_rules<typename T::value_type>;
    } else {
        return check_struct<T>();
    }
}

inline constexpr std::size_t kMaxRecordTypes = 64;

// A record type as the decoder matches call sites to it: data_type names a struct Data and is empty otherwise, and
// metadata_type is empty when any metadata goes.
struct Kind {
    std::string_view data_type, metadata_type;
    std::span<const uint16_t> data_codes, metadata_codes;
    std::span<const ValueType> value_types;
    std::size_t size = 0;
    void (*build)(void* at, const Source& source, const Timestamp& time, FieldReader& data) = nullptr;
};
void visit_stretches(
    const BatchData& batch, const std::function<void(std::size_t type, std::size_t from, std::size_t count)>& visit);

// Each call site's metadata, by zone id, which build_site_metadata builds the first time a record asks for it.
extern std::atomic<const void*> site_metadata[];
const void* build_site_metadata(uint32_t id, const void* (*build)(FieldReader& fields));
template <typename M>
const M& metadata_of(uint32_t id) {
    const void* metadata = site_metadata[id].load(std::memory_order_acquire);
    if (metadata == nullptr) {
        metadata = build_site_metadata(id, [](FieldReader& fields) -> const void* { return new M(build<M>(fields)); });
    }
    return *static_cast<const M*>(metadata);
}

template <typename R, typename... Ts>
Batch<Ts...> batch_parameter(std::function<R(const Batch<Ts...>&)>);
template <typename R, typename... Ts>
Batch<Ts...> batch_parameter(std::function<R(Batch<Ts...>)>);
template <typename F>
using CallbackBatch = decltype(batch_parameter(std::function{std::declval<std::unwrap_ref_decay_t<F>>()}));
}  // namespace detail

/**
 * @brief A record from a core: Payload, sent with Metadata.
 *
 * payload() reaches the payload. Arguments<T> takes any arguments that are all T, or all among T's alternatives when T
 * is a std::variant, and payload() is their span. void means no payload. Record<Payload> takes Payload whatever its
 * metadata.
 */
template <typename Payload, typename Metadata>
class Record {
public:
    /** @brief The record's payload: its struct, or its arguments' span. */
    const auto& payload() const
        requires(!std::is_void_v<Payload>)
    {
        if constexpr (detail::is_arguments<Payload>) {
            return payload_.get();
        } else {
            return payload_;
        }
    }

    /** @brief The core that emitted the record. */
    Core core() const {
        return Core{
            .logical = CoreCoord(source_.logical_x, source_.logical_y),
            .physical = CoreCoord(source_.physical_x, source_.physical_y),
            .chip_id = source_.chip_id,
            .processor = source_.processor};
    }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const { return source_.runtime_id; }
    /** @brief The location of the call that published the record. */
    const SourceLocation& location() const { return detail::location_of(source_.zone_id); }

    /** @brief When the core published the record. */
    const Timestamp& published_at() const { return time_; }

    /** @brief The record's metadata. */
    const auto& metadata() const
        requires(!std::same_as<Metadata, detail::AnyMetadata> && !std::is_void_v<Metadata>)
    {
        return detail::metadata_of<Metadata>(source_.zone_id);
    }

private:
    template <RecordType... Ts>
        requires(sizeof...(Ts) > 0)
    friend class Batch;
    // The payload is built in place, so the record's padding keeps what its storage held.
    Record(const detail::Source& source, const Timestamp& time, detail::FieldReader& data) :
        source_(source), time_(time), payload_(build_payload(data)) {}
    static typename detail::Stored<Payload>::type build_payload(detail::FieldReader& data) {
        if constexpr (detail::is_arguments<Payload>) {
            return typename detail::Stored<Payload>::type(data);
        } else if constexpr (std::is_void_v<Payload>) {
            return {};
        } else {
            return detail::build<Payload>(data);
        }
    }
    static constexpr detail::Kind kind() {
        // A type that breaks a rule is that rule's one error: nothing below is built for it.
        detail::Kind kind{};
        if constexpr (detail::follows_rules<Payload>() && detail::follows_rules<Metadata>()) {
            static_assert(
                std::ranges::find(detail::leaf_codes<Metadata>, detail::type_code<Timestamp>()) ==
                    detail::leaf_codes<Metadata>.end(),
                "Timestamps in metadata are not supported");
            static_assert(
                alignof(Record) <= alignof(std::max_align_t),
                "alignment beyond alignof(std::max_align_t) is not supported");
            kind = {
                .data_type = std::is_class_v<Payload> && !detail::is_arguments<Payload> ? detail::type_name<Payload>
                                                                                        : std::string_view(),
                .metadata_type =
                    std::is_same_v<Metadata, detail::AnyMetadata>
                        ? std::string_view()
                        : detail::type_name<std::conditional_t<std::is_void_v<Metadata>, detail::NoMetadata, Metadata>>,
                .data_codes = detail::leaf_codes<Payload>,
                .metadata_codes = detail::leaf_codes<Metadata>,
                .value_types = detail::value_types<Payload>,
                .size = sizeof(Record),
                .build = [](void* at, const detail::Source& source, const Timestamp& time, detail::FieldReader& data) {
                    new (at) Record(source, time, data);
                }};
        }
        return kind;
    }

    detail::Source source_;
    Timestamp time_;
    [[no_unique_address]] typename detail::Stored<Payload>::type payload_;
};

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
        std::string name, std::span<const detail::Kind> kinds, std::function<void(const detail::BatchData&)> callback);
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
        detail::CallbackBatch<F>::kKinds,
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
    static_assert(sizeof...(Ts) <= detail::kMaxRecordTypes, "a callback can take at most 64 record types");

public:
    /** @brief The batch's records of type T. */
    template <typename T>
        requires(std::same_as<T, Ts> || ...)
    std::span<const T> records() const {
        const std::size_t count = data_.counts[kIndex<T>];
        return count == 0 ? std::span<const T>()
                          : std::span(std::launder(reinterpret_cast<const T*>(data_.records[kIndex<T>])), count);
    }
    /** @brief Calls f with each of the batch's records, each processor's in the order it emitted them. */
    template <typename F>
        requires(std::invocable<F&, const Ts&> && ...)
    void visit(F&& f) const {
        detail::visit_stretches(data_, [&](std::size_t type, std::size_t from, std::size_t count) {
            (visit_stretch<Ts>(type, from, count, f), ...);
        });
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
    static constexpr std::array<detail::Kind, sizeof...(Ts)> kKinds{Ts::kind()...};
    template <typename T>
    static constexpr std::size_t kIndex = std::variant<std::type_identity<Ts>...>(std::type_identity<T>()).index();
    template <typename T, typename F>
    void visit_stretch(std::size_t type, std::size_t from, std::size_t count, F& f) const {
        if (type == kIndex<T>) {
            for (const T& record : records<T>().subspan(from, count)) {
                f(record);
            }
        }
    }
    explicit Batch(const detail::BatchData& data) : data_(data) {}

    const detail::BatchData& data_;
};

/** @brief Returns true if the streaming profiler is currently running on at least one device. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
