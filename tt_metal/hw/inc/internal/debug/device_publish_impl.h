// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish's implementation (api/debug/device_publish.h): its options, the rules a call's types follow, Strings,
// and each call site with the descriptor the host reads it by. A call site encodes its records by
// internal/debug/device_publish_encoding.h.
//
// The metadata never reaches the device. The device compiler copies it, the source location and the type names into
// sections the device never loads, next to one descriptor per call site (.device_publish_descriptors). The call site's
// id is the address of a one-byte handle in .device_publish_ids, which the host rebases into the process's id space
// when it loads the ELF (llrt/call_sites.cpp).
#pragma once

#include <type_traits>
#include <utility>

#include "internal/debug/device_publish_encoding.h"

// The sections a call site's objects go to, which the linker scripts keep out of the device image and
// llrt/call_sites.cpp reads.
#define DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME ".device_publish_descriptor_contents"
#define DEVICE_PUBLISH_DESCRIPTORS_SECTION_NAME ".device_publish_descriptors"
#define DEVICE_PUBLISH_STRINGS_SECTION_NAME ".device_publish_strings"
#define DEVICE_PUBLISH_IDS_SECTION_NAME ".device_publish_ids"

namespace tt::tt_metal::experimental::device_observability {
/** @brief The DevicePublish option that lets a record wait for its RISC's next batch of records, or the launch's end,
 * before the host sees it, which saves a fence per record. It doesn't combine with kSendToExternalProcess, whose
 * records are published at once. */
inline constexpr struct PublishBatched {
} kPublishBatched{};
/** @brief The DevicePublish option that also sends the record to the external process, a process on this host that
 * registered callbacks without opening devices of its own. The Metal process gets it too. */
inline constexpr struct SendToExternalProcess {
} kSendToExternalProcess{};
/** @brief The DevicePublish option that writes the record out of line even when DevicePublish is inlined, so the call
 * site costs only a call. */
inline constexpr struct NoInline {
} kNoInline{};
}  // namespace tt::tt_metal::experimental::device_observability

namespace tt::tt_metal::experimental::device_observability::detail {
// An internal DevicePublish option, for the tools built on it: the record leaves out the Timestamp field kField names,
// and the host fills it in with the record's publish time. The call passes one struct, the one kField belongs to:
// DevicePublish<Metadata{...}, kPublishTimeIn<&Interval::end>>(Interval{start}) sends only the start.
template <auto kField>
struct PublishTimeIn {};
template <auto kField>
inline constexpr PublishTimeIn<kField> kPublishTimeIn{};
}  // namespace tt::tt_metal::experimental::device_observability::detail

namespace device_observability {

template <typename T>
inline constexpr bool is_publish_time_option = false;
template <auto kField>
inline constexpr bool is_publish_time_option<detail::PublishTimeIn<kField>> = true;

// Whether T is one of DevicePublish's options rather than metadata.
template <typename T>
inline constexpr bool is_publish_option =
    std::is_same_v<std::remove_cv_t<T>, PublishBatched> || std::is_same_v<std::remove_cv_t<T>, SendToExternalProcess> ||
    std::is_same_v<std::remove_cv_t<T>, NoInline> || is_publish_time_option<std::remove_cv_t<T>>;

template <typename... Options>
constexpr bool check_options() {
    constexpr bool known = (is_publish_option<Options> && ...);
    static_assert(
        known, "unknown DevicePublish option; the options are kPublishBatched, kSendToExternalProcess and kNoInline");
    constexpr bool batched_and_external = (std::is_same_v<std::remove_cv_t<Options>, PublishBatched> || ...) &&
                                          (std::is_same_v<std::remove_cv_t<Options>, SendToExternalProcess> || ...);
    static_assert(
        !batched_and_external,
        "kPublishBatched doesn't combine with kSendToExternalProcess, whose records are published at once");
    return known && !batched_and_external;
}
template <typename Option, auto... Options>
inline constexpr bool kHasOption = (std::is_same_v<std::remove_cv_t<decltype(Options)>, Option> || ...);

template <typename Payload, size_t... I>
constexpr bool check_arguments(std::index_sequence<I...>) {
    return (detail::check_argument<ArgumentAt<Payload, I>>() & ... & true);
}
// The rules a DevicePublish call's types follow, given its arguments' types as forwarding references deduce them, with
// the profiler on or off; true when they all hold. Each mistake is one error.
template <typename Metadata, typename... Args>
constexpr bool check_call() {
    using Payload = RecordPayload<Args...>;
    constexpr bool metadata = detail::check_metadata<Metadata>();
    constexpr bool data = check_arguments<Payload>(ArgumentIndices<Payload>{});
    if constexpr (metadata && data) {
        if constexpr (detail::holds<Payload>(detail::kStringCode)) {
            static_assert(
                !detail::holds<Payload>(detail::kStringCode),
                "strings are not supported in a payload; send the text as metadata, a DevicePublish call per string. A "
                "const char array counts as a string; send bytes in a std::array");
            return false;
        } else {
            // A record's longest packet, its full one, must fit the ring's room for one. It is never shorter than a
            // medium one with the timer packet ahead of it.
            static_assert(
                kLayout<Payload, detail::TimestampWidth::Bits64>.packet_words <= ppfmt::MAX_RECORD_WORDS,
                "this record is over the 2004-byte limit; split it into several records");
        }
    }
    return metadata && data;
}

// Every object below is a static template, so a call site in an inline function, a template or a class-template member
// is never COMDAT, which GCC can't put in a named section. Nothing is `used`: the call site's asm references its handle
// and descriptor, so a copy of an inline function that LTO drops takes its descriptor with it.

// A string's exact-length copy, one per distinct text.
template <typename C, C... cs>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static constexpr C section_string[] = {
    cs..., C(0)};

// A text's 64-bit FNV-1a hash, which a String holds.
template <typename C, C... cs>
inline constexpr uint64_t kStringHash = [] {
    uint64_t hash = 0xcbf29ce484222325;
    ((hash = (hash ^ uint8_t(cs)) * 0x100000001b3), ...);
    return hash;
}();
// A String's text by its hash, for the host to look it up by. A String doesn't point at it, so nothing would keep it
// but `used`.
template <typename C, C... cs>
[[gnu::section(DEVICE_PUBLISH_STRINGS_SECTION_NAME),
  gnu::used]] static const detail::StringEntry section_string_entry = {
    kStringHash<C, cs...>, section_string<C, cs...>, uint32_t(sizeof...(cs))};

template <typename T>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static const auto section_type_name =
    detail::kTypeNameChars<T>;

// The size of a call site's function name: its __PRETTY_FUNCTION__ without the " [with ...]" GCC appends inside a
// template, so every instantiation of a template is one call site.
constexpr size_t function_name_size(const char* pretty) {
    const char* with = __builtin_strstr(pretty, " [with ");
    return (with ? static_cast<size_t>(with - pretty) : __builtin_strlen(pretty)) + 1;
}
template <size_t N>
struct CallSiteLocation {
    constexpr CallSiteLocation(String file, uint32_t line, const char* pretty) : file(file), line(line) {
        for (size_t k = 0; k + 1 < N; ++k) {
            function[k] = pretty[k];
        }
    }
    String file;
    uint32_t line;
    char function[N] = {};
};
template <auto kLocation, size_t... I>
constexpr const char* section_function_name(std::index_sequence<I...>) {
    return section_string<char, kLocation.function[I]...>;
}

template <auto kMetadata>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static const auto section_metadata =
    kMetadata;

template <typename T>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static const auto section_codes =
    detail::leaf_type_codes<T>;

template <typename T>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static const auto section_field_bits =
    kFieldBits<T>;

// A call's payload: the one its arguments make, or with kPublishTimeIn<kField>, that struct less the field.
template <typename Arguments, typename... Options>
struct CallPayloadOf {
    using type = Arguments;
};
template <typename Arguments, typename Option, typename... Rest>
struct CallPayloadOf<Arguments, Option, Rest...> : CallPayloadOf<Arguments, Rest...> {};
template <typename Arguments, auto kField, typename... Rest>
struct CallPayloadOf<Arguments, detail::PublishTimeIn<kField>, Rest...> {
    static_assert(
        std::is_same_v<Arguments, typename MemberOf<decltype(kField)>::ClassType> &&
            std::is_same_v<typename MemberOf<decltype(kField)>::MemberType, Timestamp>,
        "kPublishTimeIn<&S::field> takes one argument, an S, and names one of its Timestamps");
    using type = PublishTimePayload<kField>;
};

template <typename V>
constexpr detail::CallArgument call_site_argument() {
    if constexpr (detail::is_std_array<V>) {
        detail::CallArgument argument = call_site_argument<typename V::value_type>();
        argument.count = uint32_t(std::tuple_size_v<V>);
        argument.is_array = 1;
        return argument;
    } else if constexpr (detail::is_leaf_type<V>) {
        const char* const name = std::is_enum_v<V> ? section_type_name<V>.data() : nullptr;
        return {.type = name, .field_count = 1, .count = 1, .is_array = 0, .field_bits = nullptr, .size = sizeof(V)};
    } else {
        return {
            .type = section_type_name<V>.data(),
            .field_count = uint32_t(detail::leaf_type_codes<V>.size()),
            .count = 1,
            .is_array = 0,
            .field_bits = section_field_bits<V>.data(),
            .size = sizeof(V)};
    }
}
template <typename Payload, size_t... I>
constexpr std::array<detail::CallArgument, sizeof...(I)> argument_table(std::index_sequence<I...>) {
    return {call_site_argument<ArgumentAt<Payload, I>>()...};
}
template <typename Payload>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static const auto section_arguments =
    argument_table<Payload>(ArgumentIndices<Payload>{});

// A form's slices for the host: each argument whole for a payload without distances, whose slices lie together; each
// slice otherwise.
template <typename Payload, detail::TimestampWidth kWidth>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME),
  maybe_unused]] static const auto section_layout_slices = [] {
    if constexpr (kSendsDeltas<Payload>) {
        return kLayout<Payload, kWidth>.slices;
    } else {
        return kArgumentPlacement<Payload>;
    }
}();
template <typename Payload, detail::TimestampWidth kWidth>
constexpr detail::PacketLayout call_site_layout() {
    return {
        .timestamp_width = kWidth,
        .packet_words = kLayout<Payload, kWidth>.packet_words,
        .slices = section_layout_slices<Payload, kWidth>.data(),
        .slice_count = uint32_t(section_layout_slices<Payload, kWidth>.size())};
}
// The forms a call site keeps, shortest first: only the full one for records that go to the external process
// (CallSiteDescriptor::sends_to_external_process says why).
template <typename Payload, bool kAlwaysBits64>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTOR_CONTENTS_SECTION_NAME), maybe_unused]] static const auto section_layouts = [] {
    constexpr bool kBits16 = !kAlwaysBits64 && kHasBits16<Payload>, kBits32 = !kAlwaysBits64;
    constexpr bool kBits64 = kAlwaysBits64 || kSendsDeltas<Payload>;
    std::array<detail::PacketLayout, kBits16 + kBits32 + kBits64> list{};
    size_t n = 0;
    if constexpr (kBits16) {
        list[n++] = call_site_layout<Payload, detail::TimestampWidth::Bits16>();
    }
    if constexpr (kBits32) {
        list[n++] = call_site_layout<Payload, detail::TimestampWidth::Bits32>();
    }
    if constexpr (kBits64) {
        list[n] = call_site_layout<Payload, detail::TimestampWidth::Bits64>();
    }
    return list;
}();

// The payload type a call site describes its records as.
template <typename Payload>
struct Described {
    using type = Payload;
};
template <auto kField>
struct Described<PublishTimePayload<kField>> {
    using type = typename MemberOf<decltype(kField)>::ClassType;
};

// The call site's id is this byte's address. It is writable so that no flag can merge it with another site's:
// -fmerge-all-constants folds identical constants, and distinct writable objects always have distinct addresses.
template <auto kLocation, auto kMetadata, typename Payload>
[[gnu::section(DEVICE_PUBLISH_IDS_SECTION_NAME), maybe_unused]] static char call_site_handle = 0;

template <auto kLocation, auto kMetadata, typename Payload, bool kSendsToExternalProcess>
[[gnu::section(DEVICE_PUBLISH_DESCRIPTORS_SECTION_NAME),
  maybe_unused]] static const detail::CallSiteDescriptor call_site_descriptor = {
    .file = kLocation.file.hash,
    .id = &call_site_handle<kLocation, kMetadata, Payload>,
    .line = kLocation.line,
    .function = section_function_name<kLocation>(std::make_index_sequence<sizeof(kLocation.function) - 1>{}),
    .metadata_type = section_type_name<decltype(kMetadata)>.data(),
    .metadata = &section_metadata<kMetadata>,
    .metadata_size = sizeof(kMetadata),
    .metadata_field_bits = section_field_bits<decltype(kMetadata)>.data(),
    .metadata_codes = section_codes<decltype(kMetadata)>.data(),
    .metadata_field_count = uint32_t(section_codes<decltype(kMetadata)>.size()),
    .payload_codes = section_codes<typename Described<Payload>::type>.data(),
    .payload_field_count = uint32_t(section_codes<typename Described<Payload>::type>.size()),
    .payload_arguments = section_arguments<typename Described<Payload>::type>.data(),
    .payload_argument_count = uint32_t(section_arguments<typename Described<Payload>::type>.size()),
    .payload_publish_time_field = uint32_t(kUnsentLeaf<Payload>),
    .packet_layouts = section_layouts<Payload, kSendsToExternalProcess>.data(),
    .packet_layout_count = uint32_t(section_layouts<Payload, kSendsToExternalProcess>.size()),
    .sends_to_external_process = kSendsToExternalProcess,
};

// A call site, unique to the file, line and function TT_DEVICE_PUBLISH_INTERNAL names: DevicePublish's entry point in
// every build. DevicePublish<Metadata{...}, kOptions...>(...), or, for a record without metadata,
// DevicePublish<kOptions...>(...) and DevicePublish<>(...), names an empty object whose call publishes the record, so a
// scope can take one and publish when it closes. Its records go on its RISC's ring, through the encoding's writer;
// elsewhere a call checks its types by the same rules and compiles to nothing. Everything that depends only on the
// types is shared between call sites, so each call site instantiates this one function.
template <auto kLocation>
struct CallSite {
    template <auto kMetadata, auto... Options, typename... Arguments>
    static FORCE_INLINE void publish_with_metadata([[maybe_unused]] Arguments&&... arguments) {
        if constexpr (check_call<decltype(kMetadata), Arguments...>() && check_options<decltype(Options)...>()) {
#if defined(DEVICE_OBSERVABILITY) && defined(PROFILE_KERNEL) && !defined(DEVICE_OBSERVABILITY_RELAY_KERNEL)
            using Payload =
                typename CallPayloadOf<RecordPayload<Arguments...>, std::remove_cv_t<decltype(Options)>...>::type;
            constexpr bool immediately = !kHasOption<PublishBatched, Options...>;
            constexpr bool external = kHasOption<SendToExternalProcess, Options...>;
            // Only an inlined writer needs keeping out of line; otherwise every call site shares one.
            constexpr bool no_inline = kHasOption<NoInline, Options...> && kWriterInlined;
            const uint32_t header = site_header<
                kMetadata,
                Payload,
                ppfmt::record_w0(kFirstPacketWords<Payload>) | (external ? ppfmt::TO_EXTERNAL : 0u)>();
            publish_record<immediately, external, no_inline, Payload>(
                header,
                record_source(arguments...),
                std::make_index_sequence<std::tuple_size_v<decltype(RegisterWords<Payload>::words)>>{});
#endif
        }
    }

    template <auto kFirst, auto... Rest>
    struct Publisher {
        template <typename... Arguments>
        FORCE_INLINE void operator()(Arguments&&... arguments) const {
            if constexpr (!is_publish_option<decltype(kFirst)>) {
                publish_with_metadata<kFirst, Rest...>(std::forward<Arguments>(arguments)...);
            } else if constexpr ((is_publish_option<decltype(Rest)> && ...)) {
                publish_with_metadata<detail::NoMetadata{}, kFirst, Rest...>(std::forward<Arguments>(arguments)...);
            } else {
                static_assert(
                    (is_publish_option<decltype(Rest)> && ...),
                    "DevicePublish takes its metadata before its options: DevicePublish<Metadata{...}, kOption>(...)");
            }
        }
    };
    template <auto kFirst = detail::NoMetadata{}, auto... Rest>
    static constexpr Publisher<kFirst, Rest...> publish{};

private:
    // Word 0 of the call site's packets: lui/addi of the handle, one pair per call site, so the loader can rebase it
    // into the call site's id. The relocation's addend carries kTypeAndLength, which the loader keeps. The asm also
    // references the descriptor, so the descriptor exists exactly when the call site's code does.
    template <auto kMetadata, typename Payload, uint32_t kTypeAndLength>
    static FORCE_INLINE uint32_t site_header() {
        uint32_t header;
        asm("lui %0, %%hi(%1+%3)\n\taddi %0, %0, %%lo(%1+%3)  # %2"
            : "=r"(header)
            : "i"(&call_site_handle<kLocation, kMetadata, Payload>),
              "i"(&call_site_descriptor < kLocation, kMetadata, Payload, (kTypeAndLength & ppfmt::TO_EXTERNAL) != 0 >),
              "i"(kTypeAndLength));
        return header;
    }
};

}  // namespace device_observability

// Without the ring a Timestamp only reaches DevicePublish, which drops it, so taking one reads no clock.
FORCE_INLINE tt::tt_metal::experimental::device_observability::Timestamp
tt::tt_metal::experimental::device_observability::Timestamp::now() {
#if defined(DEVICE_OBSERVABILITY) && defined(PROFILE_KERNEL) && !defined(DEVICE_OBSERVABILITY_RELAY_KERNEL)
    uint32_t high, low;
    ::device_observability::read_wall_clock(high, low);
    return Timestamp(uint64_t{high} << 32 | low);
#else
    return {};
#endif
}

// Usable as a template argument (a string literal is not). The String is its text's hash, and taking the text's entry
// keeps the text in the image, where the host looks it up by that hash.
template <typename C, C... cs>
constexpr tt::tt_metal::experimental::device_observability::String operator""_s() {
    [[maybe_unused]] const auto* const entry = &::device_observability::section_string_entry<C, cs...>;
    tt::tt_metal::experimental::device_observability::String string;
    string.hash = ::device_observability::kStringHash<C, cs...>;
    return string;
}

// Appends _s to a literal after the argument expands, so a tool's macro can take a plain literal, a macro that expands
// to one, or #__VA_ARGS__.
#define TT_RECORD_STRING_(literal) literal##_s
#define TT_RECORD_STRING(literal) TT_RECORD_STRING_(literal)

// DevicePublish (api/debug/device_publish.h): the publish object of the call site on the line where it expands. Its
// <...> sets the metadata and options and its (...) calls it.
#define TT_DEVICE_PUBLISH_INTERNAL                                                                                 \
    ::device_observability::CallSite<                                                                              \
        ::device_observability::CallSiteLocation<::device_observability::function_name_size(__PRETTY_FUNCTION__)>{ \
            TT_RECORD_STRING(__FILE__), __LINE__, __PRETTY_FUNCTION__}>::template publish
