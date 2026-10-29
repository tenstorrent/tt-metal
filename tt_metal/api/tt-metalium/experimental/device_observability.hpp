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
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt_stl/strong_type.hpp>
#include <hostdevcommon/device_observability_reflection.hpp>
#include <hostdevcommon/device_observability_types.hpp>

// Device observability's host API. A callback registered here receives the records device kernels publish with
// DevicePublish (api/debug/device_publish.h), and the tools built on it, like the streaming profiler, publish theirs
// the same way.
//
// This API is experimental and may change or be removed without notice.
//
//     Callback callback = RegisterCallback(
//         [](const Batch<Record<TileStats, Name>>& batch) {
//             for (const auto& record : batch.records<Record<TileStats, Name>>()) {
//                 use(record.metadata().name, record.core().logical, record.payload().tiles);
//             }
//         },
//         "my-tool");
//
namespace tt::tt_metal::experimental::device_observability {

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

/**
 * @brief The payload of a record whose calls pass any number of arguments (including none), each a T:
 * `DevicePublish<M>(a, b, ...)`. payload() is their std::span<const T>.
 */
template <typename T>
struct Variadic {
    using value_type = T;
};

class Callback;

template <typename Payload, typename Metadata>
class Record;

/** @brief A type a Batch can hold: a Record. */
template <typename T>
concept RecordType = detail::is_specialization<T, Record>;

template <RecordType... Ts>
    requires(sizeof...(Ts) > 0)
class Batch;

/** @brief A stall: the time a core spent waiting for its records to drain. */
using Stall = Record<Interval, StallMetadata>;

// Implementation detail.
namespace detail {
// A Metal process's call sites, by the ids its images gave them. Its receiver owns them (tt::llrt).
class CallSites;
const SourceLocation& location_of(const CallSites& call_sites, uint32_t call_site_id);

struct BatchData {
    uint64_t dropped_bytes = 0;
    uint64_t stall_count = 0;
    std::span<const std::byte* const> records;
    std::span<const std::size_t> counts;
    std::span<const uint32_t> order;
};
struct RecordTypeDescriptor;
using CallbackId = ttsl::StrongType<uint32_t, struct CallbackIdTag>;
Callback register_callback(
    std::string name,
    std::span<const RecordTypeDescriptor> record_types,
    std::function<void(const BatchData&)> callback,
    bool external_process);

class FieldReader {
public:
    virtual ~FieldReader() = default;
    template <typename T>
    T leaf() {
        T value{};
        read(std::as_writable_bytes(std::span(&value, 1)));
        return value;
    }

    // A record's arguments: each one's alternative and field count, and the arena their arrays go in.
    const std::span<const uint8_t> alternatives;
    const std::span<const uint32_t> argument_field_counts;
    std::pmr::memory_resource* const arena;

protected:
    explicit FieldReader(
        std::span<const uint8_t> alternatives = {},
        std::span<const uint32_t> argument_field_counts = {},
        std::pmr::memory_resource* arena = nullptr) :
        alternatives(alternatives), argument_field_counts(argument_field_counts), arena(arena) {}
    virtual void read(std::span<std::byte> out) = 0;
};

template <typename T>
inline constexpr bool is_array_value = false;
template <typename E>
inline constexpr bool is_array_value<std::span<const E>> = true;

// A capture's lanes, which its receiver and every kept copy of their records share: each holds a reference, and the
// last to let go deletes them.
class CaptureLanes;
void retain(const CaptureLanes& lanes);
void release(const CaptureLanes& lanes);
// A lane records come from: the core whose RISC it is, and the call sites of the Metal process that runs it.
struct RecordLane {
    Core core;
    const CallSites* call_sites = nullptr;
    // The call sites' metadata, by id, which build_site_metadata builds the first time a record asks for it.
    std::atomic<const void*>* metadata = nullptr;
    const CaptureLanes* lanes = nullptr;  // which it belongs to
};
struct RecordOrigin {
    const RecordLane* lane = nullptr;
    uint32_t call_site_id = 0;
    uint32_t runtime_id = 0;
};
// Whether a payload holds an array in the batch's arena, which a kept record copies.
template <typename T>
inline constexpr bool holds_array = is_array_value<T>;
template <typename... Ts>
inline constexpr bool holds_array<std::variant<Ts...>> = (holds_array<Ts> || ...);
template <typename... Ts>
inline constexpr bool holds_array<std::tuple<Ts...>> = (holds_array<Ts> || ...);

// A payload whose arrays sit in the batch's arena. A copy owns its arrays, so it outlives the batch.
template <typename T>
class OwnedArrays {
public:
    explicit OwnedArrays(const T& value) : value_(value) {}
    OwnedArrays(const OwnedArrays& other) : value_(other.value_) { own(value_); }
    OwnedArrays(OwnedArrays&&) noexcept = default;
    OwnedArrays& operator=(const OwnedArrays& other) { return *this = OwnedArrays(other); }
    OwnedArrays& operator=(OwnedArrays&&) noexcept = default;

    const T& get() const { return value_; }

private:
    template <typename A>
    void own(A& value) {
        if constexpr (is_array_value<A>) {
            using E = typename A::value_type;
            auto elements = std::make_shared<E[]>(value.size());
            std::ranges::copy(value, elements.get());
            for (std::size_t k = 0; k < value.size(); k++) {
                own(elements[k]);
            }
            value = A(elements.get(), value.size());
            arrays_.push_back(std::move(elements));
        } else if constexpr (is_specialization<A, std::variant>) {
            std::visit([&](auto& alternative) { own(alternative); }, value);
        } else if constexpr (is_specialization<A, std::tuple>) {
            std::apply([&](auto&... elements) { (own(elements), ...); }, value);
        }
    }

    T value_;
    std::vector<std::shared_ptr<const void>> arrays_;
};

// How a record holds its payload, and what payload() returns for it (Value).
template <typename P>
struct Stored {
    using Value = P;
    using type = std::conditional_t<holds_array<Value>, OwnedArrays<Value>, Value>;
};
template <typename E>
struct Stored<Variadic<E>> {
    using Value = std::span<const E>;
    using type = OwnedArrays<Value>;
};
template <>
struct Stored<void> {
    using type = std::monostate;
};

// Whether a payload lies the same way in every record, with no alternative to pick or array to copy, so the decoder
// writes it directly.
template <typename P>
inline constexpr bool fixed_layout = !is_array_value<P> && !is_specialization<P, std::variant>;
template <typename... Ts>
inline constexpr bool fixed_layout<std::tuple<Ts...>> = (fixed_layout<Ts> && ...);
template <typename E>
inline constexpr bool fixed_layout<Variadic<E>> = false;

// Argument k of a record's call as an A: an array's elements go in the batch's arena.
template <typename A>
A build_alternative(FieldReader& data, std::size_t k) {
    if constexpr (is_array_value<A>) {
        using E = typename A::value_type;
        const std::size_t count = data.argument_field_counts[k] / leaf_type_codes<E>.size();
        E* elements = std::pmr::polymorphic_allocator<E>(data.arena).allocate(count);
        for (std::size_t i = 0; i < count; i++) {
            std::construct_at(elements + i, build<E>(data));
        }
        return A(elements, count);
    } else {
        return build<A>(data);
    }
}
// Argument k of a record's call as an E, which takes the alternative the call site matched when E is a std::variant.
template <typename E>
E build_argument(FieldReader& data, std::size_t k) {
    if constexpr (is_specialization<E, std::variant>) {
        E value;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            static_cast<void>(
                ((data.alternatives[k] == I &&
                  (value.template emplace<I>(build_alternative<std::variant_alternative_t<I, E>>(data, k)), true)) ||
                 ...));
        }(std::make_index_sequence<std::variant_size_v<E>>());
        return value;
    } else {
        return build_alternative<E>(data, k);
    }
}
template <typename P>
typename Stored<P>::type build_payload(FieldReader& data) {
    using Result = typename Stored<P>::type;
    if constexpr (std::is_void_v<P>) {
        return {};
    } else if constexpr (is_specialization<P, Variadic>) {
        using E = typename P::value_type;
        const std::size_t count = data.alternatives.size();
        E* values = count == 0 ? nullptr : std::pmr::polymorphic_allocator<E>(data.arena).allocate(count);
        for (std::size_t k = 0; k < count; k++) {
            std::construct_at(values + k, build_argument<E>(data, k));
        }
        return Result(std::span<const E>(values, count));
    } else if constexpr (is_specialization<P, std::tuple>) {
        return Result([&]<std::size_t... I>(std::index_sequence<I...>) {
            return P{build_argument<std::tuple_element_t<I, P>>(data, I)...};  // evaluated left to right
        }(std::make_index_sequence<std::tuple_size_v<P>>()));
    } else {
        return Result(build_argument<P>(data, 0));
    }
}

struct ArgumentType {
    std::string_view name;
    std::span<const uint16_t> codes;
    bool array = false;
    std::size_t count = 0;  // an array's elements, or 0 for any number
};
template <typename T>
inline constexpr ArgumentType argument_type{
    is_leaf_type<T> && !std::is_enum_v<T> ? std::string_view() : type_name<T>, leaf_type_codes<T>};
template <typename E>
inline constexpr ArgumentType argument_type<std::span<const E>>{argument_type<E>.name, leaf_type_codes<E>, true};
template <typename E, std::size_t N>
inline constexpr ArgumentType argument_type<std::array<E, N>>{argument_type<E>.name, leaf_type_codes<E>, true, N};
// The types an argument may have: a std::variant's alternatives, or the one type.
template <typename E>
inline constexpr std::array<ArgumentType, 1> alternatives_of{argument_type<E>};
template <typename... Ts>
inline constexpr std::array<ArgumentType, sizeof...(Ts)> alternatives_of<std::variant<Ts...>>{argument_type<Ts>...};
// The alternatives of each argument a call passes, in order. A Variadic's one entry is every argument's.
template <typename P>
inline constexpr std::array<std::span<const ArgumentType>, 1> argument_positions{alternatives_of<P>};
template <>
inline constexpr std::array<std::span<const ArgumentType>, 0> argument_positions<void>{};
template <typename... Es>
inline constexpr std::array<std::span<const ArgumentType>, sizeof...(Es)> argument_positions<std::tuple<Es...>>{
    alternatives_of<Es>...};
template <typename E>
inline constexpr std::array<std::span<const ArgumentType>, 1> argument_positions<Variadic<E>>{alternatives_of<E>};

// A callback's payload and metadata types follow the rules kernels check, so a type no kernel could send fails here.
template <typename V>
struct DeviceArgument {
    using type = V;
};
template <typename E>
struct DeviceArgument<std::span<const E>> {
    using type = std::array<typename DeviceArgument<E>::type, 1>;
};
// One type an argument may have: one a kernel can send, and not a Variadic, which is only ever a whole payload.
template <typename A>
constexpr bool alternative_follows_rules() {
    static_assert(
        !is_specialization<A, Variadic>, "Variadic<T> is a whole payload; it can't be part of a tuple or variant");
    if constexpr (is_specialization<A, Variadic>) {
        return false;
    } else {
        return check_argument<typename DeviceArgument<A>::type>();
    }
}
template <typename E>
inline constexpr bool argument_follows_rules = alternative_follows_rules<E>();
template <typename... Es>
inline constexpr bool argument_follows_rules<std::variant<Es...>> = (alternative_follows_rules<Es>() & ... & true);
template <typename P>
inline constexpr bool payload_follows_rules = argument_follows_rules<P>;
template <>
inline constexpr bool payload_follows_rules<void> = true;
template <typename... Es>
inline constexpr bool payload_follows_rules<std::tuple<Es...>> = (argument_follows_rules<Es> & ... & true);
template <typename E>
inline constexpr bool payload_follows_rules<Variadic<E>> = argument_follows_rules<E>;
// Whether a payload holds a String, which only metadata can.
template <typename Payload>
constexpr bool holds_string() {
    return std::ranges::any_of(argument_positions<Payload>, [](std::span<const ArgumentType> alternatives) {
        return std::ranges::any_of(alternatives, [](const ArgumentType& type) {
            return std::ranges::find(type.codes, type_code<String>()) != type.codes.end();
        });
    });
}
template <typename Payload>
constexpr std::span<const uint16_t> payload_codes() {
    if constexpr (std::is_void_v<Payload> || !fixed_layout<Payload>) {
        return {};
    } else {
        return leaf_type_codes<Payload>;
    }
}

inline constexpr std::size_t kMaxRecordTypes = 64;
// How many of Ts are T.
template <typename T, typename... Ts>
inline constexpr std::size_t kCount = (std::size_t{std::is_same_v<T, Ts>} + ... + 0);

// A record type as the decoder matches call sites to it: by its metadata type and the types each argument may take.
// payload_codes are a fixed-layout payload's fields.
struct RecordTypeDescriptor {
    std::string_view metadata_type;
    std::span<const uint16_t> payload_codes, metadata_codes;
    std::span<const std::span<const ArgumentType>> arguments;
    bool variadic = false;
    bool fixed_layout = true;
    std::size_t size = 0;
    void (*build)(std::byte* at, const RecordOrigin& origin, const Timestamp& time, FieldReader& data) = nullptr;
};
void visit_chunks(
    const BatchData& batch, const std::function<void(std::size_t type, std::size_t from, std::size_t count)>& visit);

// Builds the metadata of a record's call site, which its call sites keep until they go, unless another call did first.
const void* build_site_metadata(
    const RecordOrigin& origin, const void* (*build)(FieldReader& fields), void (*destroy)(const void* metadata));
template <typename M>
const M& metadata_of(const RecordOrigin& origin) {
    const void* metadata = origin.lane->metadata[origin.call_site_id].load(std::memory_order_acquire);
    if (metadata == nullptr) {
        metadata = build_site_metadata(
            origin,
            [](FieldReader& fields) -> const void* { return new M(build<M>(fields)); },
            [](const void* built) { delete static_cast<const M*>(built); });
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
 * Payload is what the call passed, which payload() returns: void for no arguments, a type T for one argument, a
 * std::tuple for two or more, one per element, or Variadic<T> for any number. An argument's type is a fixed-width
 * scalar, an enum, a struct, a std::span<const E> for an array of any length, a std::array<E, N> for one of exactly N,
 * or a std::variant of those for an argument of any of them. void as Metadata means a record sent without metadata.
 */
template <typename Payload, typename Metadata>
class Record {
public:
    /** @brief The record's payload. */
    const auto& payload() const
        requires(!std::is_void_v<Payload>)
    {
        if constexpr (detail::holds_array<typename detail::Stored<Payload>::Value>) {
            return payload_.get();
        } else {
            return payload_;
        }
    }

    /** @brief The core that emitted the record. */
    Core core() const { return origin_.lane->core; }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const { return origin_.runtime_id; }
    /** @brief The location of the call that published the record. */
    const SourceLocation& location() const {
        return detail::location_of(*origin_.lane->call_sites, origin_.call_site_id);
    }

    /** @brief When the core published the record. */
    const Timestamp& published_at() const { return time_; }

    /** @brief The record's metadata. */
    const auto& metadata() const
        requires(!std::is_void_v<Metadata>)
    {
        return detail::metadata_of<Metadata>(origin_);
    }

    // A copy, unlike a record in its batch, holds a reference on its capture's lanes, so its core, location and
    // metadata stay valid for as long as it is kept.
    Record(const Record& other) : origin_(other.origin_), time_(other.time_), payload_(other.payload_) {
        detail::retain(*origin_.lane->lanes);
    }
    Record& operator=(const Record& other) {
        if (this != &other) {
            detail::retain(*other.origin_.lane->lanes);
            detail::release(*origin_.lane->lanes);
            origin_ = other.origin_;
            time_ = other.time_;
            payload_ = other.payload_;
        }
        return *this;
    }
    // A record in its batch is never destroyed, only a copy, which lets its reference go.
    ~Record() { detail::release(*origin_.lane->lanes); }

private:
    template <RecordType... Ts>
        requires(sizeof...(Ts) > 0)
    friend class Batch;
    // The payload is built in place, so the record's padding keeps what its storage held.
    Record(const detail::RecordOrigin& origin, const Timestamp& time, detail::FieldReader& data) :
        origin_(origin), time_(time), payload_(detail::build_payload<Payload>(data)) {}
    static constexpr detail::RecordTypeDescriptor descriptor() {
        // A type that breaks a rule is that rule's one error: nothing below is built for it. A record without metadata
        // comes from call sites that send NoMetadata.
        using SentMetadata = std::conditional_t<std::is_void_v<Metadata>, detail::NoMetadata, Metadata>;
        detail::RecordTypeDescriptor descriptor{};
        if constexpr (detail::payload_follows_rules<Payload> && detail::check_metadata<SentMetadata>()) {
            static_assert(
                !detail::holds<SentMetadata>(detail::type_code<Timestamp>()),
                "Timestamps in metadata are not supported");
            static_assert(
                !detail::is_specialization<Payload, std::tuple> || detail::argument_positions<Payload>.size() >= 2,
                "a std::tuple payload takes two or more arguments; use void for none and the type itself for one");
            static_assert(
                !detail::holds_string<Payload>(),
                "strings are not supported in a payload; send the text as metadata, a DevicePublish call per string. A "
                "const char array counts as a string; send bytes in a std::array");
            static_assert(
                alignof(Record) <= alignof(std::max_align_t),
                "alignment beyond alignof(std::max_align_t) is not supported");
            descriptor = {
                .metadata_type = detail::type_name<SentMetadata>,
                .payload_codes = detail::payload_codes<Payload>(),
                .metadata_codes = detail::leaf_type_codes<SentMetadata>,
                .arguments = detail::argument_positions<Payload>,
                .variadic = detail::is_specialization<Payload, Variadic>,
                .fixed_layout = detail::fixed_layout<Payload>,
                .size = sizeof(Record),
                .build = [](std::byte* at,
                            const detail::RecordOrigin& origin,
                            const Timestamp& time,
                            detail::FieldReader& data) { new (at) Record(origin, time, data); }};
        }
        return descriptor;
    }

    detail::RecordOrigin origin_;
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
        std::string name,
        std::span<const detail::RecordTypeDescriptor> record_types,
        std::function<void(const detail::BatchData&)> callback,
        bool external_process);
    explicit Callback(detail::CallbackId id) : id_(id) {}
    detail::CallbackId id_;
};

/**
 * @brief Registers a callback to be invoked when records arrive from a device.
 *
 * Multiple callbacks can be registered; each runs on its own thread, one invocation at a time. If a callback shares a
 * resource with other callbacks, access it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep
 * up with incoming data miss records; this is reported by Batch::dropped_bytes. TT_METAL_DEVICE_OBSERVABILITY_FIFO_MB
 * sets how far a callback can fall behind before it misses records. May be called before, during or between captures.
 *
 * @param callback Takes one `const Batch<Ts...>&`; Ts selects the record types it receives.
 * @param name Optional name for the callback in logs and thread names.
 * @param external_process Makes this process the external process, whose callbacks receive the records that Metal
 * processes on this host, run by any user, send with kSendToExternalProcess. It listens in
 * /tmp/tt_metal_device_observability, which a process in a container mounts. Throws when another process is the
 * external process.
 * @return The registration; the callback runs until it is destroyed.
 */
template <BatchCallable F>
Callback RegisterCallback(F callback, std::string name = {}, bool external_process = false) {
    return detail::register_callback(
        std::move(name),
        detail::CallbackBatch<F>::kRecordTypes,
        [callback = std::move(callback)](const detail::BatchData& data) mutable {
            callback(detail::CallbackBatch<F>(data));
        },
        external_process);
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
    static_assert(((detail::kCount<Ts, Ts...> == 1) && ...), "a Batch lists each record type once");

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
        detail::visit_chunks(data_, [&](std::size_t type, std::size_t from, std::size_t count) {
            (visit_chunk<Ts>(type, from, count, f), ...);
        });
    }
    /**
     * @brief Bytes of device output lost since the callback last ran.
     *
     * Nonzero if the callback could not keep up with incoming data. Raising TT_METAL_DEVICE_OBSERVABILITY_FIFO_MB lets
     * a callback fall further behind before it loses data.
     */
    uint64_t dropped_bytes() const { return data_.dropped_bytes; }
    /** @brief How many times the cores in this batch waited for their records to drain. */
    uint64_t stall_count() const { return data_.stall_count; }

private:
    template <BatchCallable F>
    friend Callback RegisterCallback(F callback, std::string name, bool external_process);
    static constexpr std::array<detail::RecordTypeDescriptor, sizeof...(Ts)> kRecordTypes{Ts::descriptor()...};
    template <typename T>
    static constexpr std::size_t kIndex = std::variant<std::type_identity<Ts>...>(std::type_identity<T>()).index();
    template <typename T, typename F>
    void visit_chunk(std::size_t type, std::size_t from, std::size_t count, F& f) const {
        if (type == kIndex<T>) {
            for (const T& record : records<T>().subspan(from, count)) {
                f(record);
            }
        }
    }
    explicit Batch(const detail::BatchData& data) : data_(data) {}

    const detail::BatchData& data_;
};

/** @brief Returns true if device observability is currently running on at least one device. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::device_observability
