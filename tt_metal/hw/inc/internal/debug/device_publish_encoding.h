// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish's encoding (internal/debug/device_publish_impl.h): what a record's packet words are and, with the
// streaming profiler on, the writer that stores a call's values as those words on its RISC's ring
// (internal/debug/device_publish_ring.h). A record's payload is laid out from its arguments' samples, the same ones the
// host reads (llrt/call_sites.cpp), as slices of each argument, which each of its packet's forms places.
#pragma once

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "hostdev/record_site.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/risc_attribs.h"

namespace kernel_profiler {
namespace streaming_profiler = tt::tt_metal::experimental::streaming_profiler;
}  // namespace kernel_profiler

namespace kernel_profiler::records {

// The type a call site describes an argument as: its own, a std::array for a C array, or a String for a string
// literal, which the payload check then rejects.
template <typename Argument>
struct ArgumentValue {
    using type = std::remove_cv_t<Argument>;
};
template <typename Element, size_t N>
struct ArgumentValue<Element[N]> {
    using type = std::array<std::remove_cv_t<Element>, N>;
};
template <size_t N>
struct ArgumentValue<const char[N]> {
    using type = streaming_profiler::String;
};
template <typename Arg>
using ValueOf = typename ArgumentValue<std::remove_reference_t<Arg>>::type;

// The payload a call's arguments make: a struct passed alone is the payload itself, and any other arguments are a tuple
// of their values.
template <typename... Args>
struct RecordPayloadOf {
    using type = std::tuple<ValueOf<Args>...>;
};
template <typename Arg>
struct RecordPayloadOf<Arg> {
    using Value = ValueOf<Arg>;
    using type = std::conditional_t<
        std::is_class_v<Value> && !streaming_profiler::detail::is_leaf_type<Value> &&
            !streaming_profiler::detail::is_std_array<Value>,
        Value,
        std::tuple<Value>>;
};
template <typename... Args>
using RecordPayload = typename RecordPayloadOf<Args...>::type;

template <typename Payload>
using ArgumentList = std::
    conditional_t<streaming_profiler::detail::is_specialization<Payload, std::tuple>, Payload, std::tuple<Payload>>;
template <typename Payload, size_t I>
using ArgumentAt = std::tuple_element_t<I, ArgumentList<Payload>>;
template <typename Payload>
using ArgumentIndices = std::make_index_sequence<std::tuple_size_v<ArgumentList<Payload>>>;

// A struct's samples, one per field: the struct with that field's bits set and every other field zero.
inline constexpr size_t kEveryField = ~size_t{0};
struct SampleSource {
    size_t field_to_set = 0;  // or kEveryField
    size_t next_field = 0;
    constexpr void open() {}
    constexpr void close() {}
    // A float's or a double's smallest negative value sets its lowest and highest bits, a bool's true its only one.
    template <typename V>
    constexpr V leaf() {
        const bool set = next_field++ == field_to_set || field_to_set == kEveryField;
        if (!set) {
            return V{};
        }
        if constexpr (
            std::is_same_v<V, streaming_profiler::String> || std::is_same_v<V, streaming_profiler::Timestamp>) {
            return __builtin_bit_cast(V, ~uint64_t{0});
        } else if constexpr (std::is_same_v<V, bool>) {
            return true;
        } else if constexpr (std::is_floating_point_v<V>) {
            return -std::numeric_limits<V>::denorm_min();
        } else if constexpr (std::is_enum_v<V>) {
            return V(~std::underlying_type_t<V>(0));
        } else {
            return V(~V(0));
        }
    }
};
template <typename T>
constexpr T sample_of(size_t field) {
    SampleSource source{field};
    return streaming_profiler::detail::build<T>(source);
}

// One byte as eight one-bit objects, so that each bit of a bit_cast is defined or not on its own.
struct ByteBits {
    unsigned char b0 : 1, b1 : 1, b2 : 1, b3 : 1, b4 : 1, b5 : 1, b6 : 1, b7 : 1;
};
template <size_t N>
struct Bytes {
    unsigned char at[N];
};
template <size_t N>
struct Bits {
    ByteBits at[N];
};
// An object's N bytes, read whole and bit by bit.
template <size_t N>
struct ObjectBytes {
    template <typename T>
    constexpr explicit ObjectBytes(const T& object) :
        bytes(__builtin_bit_cast(Bytes<N>, object)), bits(__builtin_bit_cast(Bits<N>, object)) {}
    // Byte i's value bits: __builtin_constant_p of a byte or a bit is 1 exactly when the language defines all of it,
    // which it does for value bits and not for padding bits.
    constexpr unsigned char value_bits(size_t i) const {
        if (__builtin_constant_p(bytes.at[i])) {
            return 0xFF;
        }
        const ByteBits& b = bits.at[i];
        return __builtin_constant_p(b.b0) | __builtin_constant_p(b.b1) << 1 | __builtin_constant_p(b.b2) << 2 |
               __builtin_constant_p(b.b3) << 3 | __builtin_constant_p(b.b4) << 4 | __builtin_constant_p(b.b5) << 5 |
               __builtin_constant_p(b.b6) << 6 | __builtin_constant_p(b.b7) << 7;
    }
    // Byte i with its padding bits zero, given its value bits.
    constexpr unsigned char defined(size_t i, unsigned char value_bits) const {
        if (value_bits == 0xFF) {
            return bytes.at[i];
        }
        const ByteBits& b = bits.at[i];
        return (value_bits & 0x01 ? b.b0 : 0) | (value_bits & 0x02 ? b.b1 : 0) << 1 |
               (value_bits & 0x04 ? b.b2 : 0) << 2 | (value_bits & 0x08 ? b.b3 : 0) << 3 |
               (value_bits & 0x10 ? b.b4 : 0) << 4 | (value_bits & 0x20 ? b.b5 : 0) << 5 |
               (value_bits & 0x40 ? b.b6 : 0) << 6 | (value_bits & 0x80 ? b.b7 : 0) << 7;
    }
    Bytes<N> bytes;
    Bits<N> bits;
};
// T's samples, each its representation as the language defines it: each value bit as the sample sets it and every
// padding bit zero, which nothing about how the compiler writes padding can change. An empty type has no samples, and
// one with unique object representations has no padding bits (the language says so), so its samples as written are
// exactly its value bits. Which of T's bits are value bits doesn't depend on the values, so the sample with every field
// set tells them.
template <typename T, size_t... K>
constexpr auto samples_of(std::index_sequence<K...>) {
    if constexpr (std::is_empty_v<T> || std::has_unique_object_representations_v<T>) {
        return std::array<T, sizeof...(K)>{sample_of<T>(K)...};
    } else {
        const ObjectBytes<sizeof(T)> full(sample_of<T>(kEveryField));
        Bytes<sizeof(T)> mask{};
        for (size_t i = 0; i < sizeof(T); ++i) {
            mask.at[i] = full.value_bits(i);
        }
        const T each[] = {sample_of<T>(K)...};
        Bytes<sizeof(T) * sizeof...(K)> samples{};
        size_t at = 0;
        for (const T& sample : each) {
            const ObjectBytes<sizeof(T)> object(sample);
            for (size_t i = 0; i < sizeof(T); ++i) {
                samples.at[at++] = object.defined(i, mask.at[i]);
            }
        }
        return samples;
    }
}

// T's samples, which its descriptor carries and its field layout comes from.
template <typename T>
inline constexpr auto kSamples =
    samples_of<T>(std::make_index_sequence<streaming_profiler::detail::leaf_type_codes<T>.size()>{});
// T's sample for field `field` as bytes, its padding zero.
template <typename T>
constexpr Bytes<sizeof(T)> sample_bytes(size_t field) {
    if constexpr (std::is_empty_v<T> || std::has_unique_object_representations_v<T>) {
        return __builtin_bit_cast(Bytes<sizeof(T)>, kSamples<T>[field]);
    } else {
        Bytes<sizeof(T)> bytes{};
        for (size_t i = 0; i < sizeof(T); ++i) {
            bytes.at[i] = kSamples<T>.at[field * sizeof(T) + i];
        }
        return bytes;
    }
}

constexpr uint32_t round_up(uint32_t bytes, uint32_t align) { return (bytes + align - 1) / align * align; }

// The leaf of an argument of type V that a record doesn't send (PublishTimeIn), or none.
template <typename V>
inline constexpr size_t kUnsentLeaf = ~size_t{0};

// Where each of V's fields lies, as its first byte and the bytes up to its last: a struct's from its samples, so they
// are where the compiler put them whatever its bit-fields, packing or alignas, and an array's its element's, repeated.
// A record sends an argument's first sent_bytes, up to the end of the last field it sends.
template <size_t kLeaves>
struct FieldLayout {
    std::array<uint16_t, kLeaves> offsets{};
    std::array<uint16_t, kLeaves> sizes{};
    size_t unsent = ~size_t{0};
    uint32_t sent_bytes = 0;
};
template <typename V>
constexpr auto lay_out_fields() {
    FieldLayout<streaming_profiler::detail::leaf_type_codes<V>.size()> layout;
    if constexpr (streaming_profiler::detail::is_leaf_type<V>) {
        layout.sizes[0] = uint16_t(sizeof(V));
    } else if constexpr (streaming_profiler::detail::is_std_array<V>) {
        using Element = typename V::value_type;
        constexpr auto element = lay_out_fields<Element>();
        constexpr size_t kElementLeaves = element.offsets.size();
        for (size_t e = 0; e < std::tuple_size_v<V>; ++e) {
            for (size_t leaf = 0; leaf < kElementLeaves; ++leaf) {
                layout.offsets[e * kElementLeaves + leaf] = uint16_t(e * sizeof(Element) + element.offsets[leaf]);
                layout.sizes[e * kElementLeaves + leaf] = element.sizes[leaf];
            }
        }
    } else {
        for (size_t field = 0; field < layout.offsets.size(); ++field) {
            const Bytes<sizeof(V)> sample = sample_bytes<V>(field);
            size_t first = sizeof(V), last = 0;
            for (size_t byte = 0; byte < sizeof(V); ++byte) {
                if (sample.at[byte] != 0) {
                    first = std::min(first, byte);
                    last = byte;
                }
            }
            layout.offsets[field] = uint16_t(first);
            layout.sizes[field] = uint16_t(last + 1 - first);
        }
    }
    layout.unsent = kUnsentLeaf<V>;
    for (size_t field = 0; field < layout.offsets.size(); ++field) {
        if (field != layout.unsent) {
            layout.sent_bytes = std::max<uint32_t>(layout.sent_bytes, layout.offsets[field] + layout.sizes[field]);
        }
    }
    return layout;
}
template <typename V>
inline constexpr auto kFieldLayout = lay_out_fields<V>();

// A record's payload is made of slices of its arguments. Each slice is 1, 2, 4 or 8 bytes at a multiple of its size
// within its argument, so it loads whole. Each field's slice is the aligned piece of its type's size that holds it: the
// field itself, or for a bit-field its storage, so a piece another holds goes. Every other byte before sent_bytes goes
// in the widest aligned pieces that fit. A field packing moved off its alignment has no piece, so a record with one
// sends its Timestamps whole, and the host reads it from the argument's bytes. The field a record is published into
// takes none.
struct Slice {
    bool is_distance;                // a Timestamp sent as its distance back from the record's time
    uint16_t argument, from, bytes;  // the argument it is from, its offset in that argument, and its size
};
template <size_t kMax>
struct ArgumentSlices {
    std::array<Slice, kMax> slices{};
    size_t count = 0;
    bool holds_every_field = true;  // whether each field it sends lies in one of its slices
};
template <typename V>
constexpr auto slice_argument() {
    constexpr auto& layout = kFieldLayout<V>;
    constexpr auto& codes = streaming_profiler::detail::leaf_type_codes<V>;
    ArgumentSlices<codes.size() + sizeof(V)> out;
    const uint32_t widest = uint32_t(std::min<size_t>(alignof(V), 8));
    std::array<uint32_t, codes.size()> piece_from{}, piece_bytes{};
    for (size_t field = 0; field < codes.size(); ++field) {
        const uint32_t from = layout.offsets[field], end = from + layout.sizes[field];
        const uint32_t bytes = std::min<uint32_t>(streaming_profiler::detail::code_bytes(codes[field]), widest);
        if (field == layout.unsent) {
            continue;
        }
        if (end <= from / bytes * bytes + bytes) {
            piece_from[field] = from / bytes * bytes;
            piece_bytes[field] = bytes;
        } else {
            out.holds_every_field = false;
        }
    }
    bool covered[sizeof(V)] = {};
    if (layout.unsent < codes.size()) {
        for (uint32_t byte = layout.offsets[layout.unsent];
             byte < layout.offsets[layout.unsent] + layout.sizes[layout.unsent];
             ++byte) {
            covered[byte] = true;
        }
    }
    const auto holds = [&](size_t outer, size_t inner) {
        return piece_from[outer] <= piece_from[inner] &&
               piece_from[inner] + piece_bytes[inner] <= piece_from[outer] + piece_bytes[outer] &&
               (piece_bytes[outer] > piece_bytes[inner] || outer < inner);
    };
    for (size_t field = 0; field < codes.size(); ++field) {
        bool held = piece_bytes[field] == 0;
        for (size_t other = 0; other < codes.size() && !held; ++other) {
            held = other != field && piece_bytes[other] != 0 && holds(other, field);
        }
        if (held) {
            continue;
        }
        const bool timestamp = codes[field] == streaming_profiler::detail::kTimestampCode;
        out.slices[out.count++] = {timestamp, 0, uint16_t(piece_from[field]), uint16_t(piece_bytes[field])};
        for (uint32_t byte = piece_from[field]; byte < piece_from[field] + piece_bytes[field]; ++byte) {
            covered[byte] = true;
        }
    }
    const auto is_free = [&](uint32_t from, uint32_t bytes) {
        for (uint32_t byte = from; byte < from + bytes; ++byte) {
            if (covered[byte]) {
                return false;
            }
        }
        return true;
    };
    for (uint32_t byte = 0; byte < layout.sent_bytes;) {
        if (covered[byte]) {
            ++byte;
            continue;
        }
        uint32_t bytes = widest;
        while (byte % bytes != 0 || byte + bytes > layout.sent_bytes || !is_free(byte, bytes)) {
            bytes /= 2;
        }
        out.slices[out.count++] = {false, 0, uint16_t(byte), uint16_t(bytes)};
        byte += bytes;
    }
    return out;
}
template <typename V>
inline constexpr auto kArgumentSlices = slice_argument<V>();

// A Timestamp travels as a distance when each argument's fields lie in its slices; otherwise it goes whole.
template <typename Payload, size_t... I>
constexpr bool sends_distances(std::index_sequence<I...>) {
    return streaming_profiler::detail::holds<Payload>(streaming_profiler::detail::kTimestampCode) &&
           (kArgumentSlices<ArgumentAt<Payload, I>>.holds_every_field && ...);
}
template <typename Payload>
inline constexpr bool kSendsDistances = sends_distances<Payload>(ArgumentIndices<Payload>{});

// Puts argument `argument`'s slices in `slices`, from slices[k] on.
template <typename V>
constexpr void argument_slices(Slice* slices, size_t& k, uint16_t argument, bool as_distances) {
    constexpr auto& own = kArgumentSlices<V>;
    for (size_t s = 0; s < own.count; ++s) {
        slices[k] = own.slices[s];
        slices[k].argument = argument;
        slices[k].is_distance = as_distances && own.slices[s].is_distance;
        ++k;
    }
}
template <typename Payload, size_t... I>
constexpr auto record_slices(std::index_sequence<I...>) {
    std::array<Slice, (kArgumentSlices<ArgumentAt<Payload, I>>.count + ... + 0)> slices{};
    [[maybe_unused]] size_t count = 0;
    (argument_slices<ArgumentAt<Payload, I>>(slices.data(), count, uint16_t(I), kSendsDistances<Payload>), ...);
    return slices;
}
template <typename Payload>
inline constexpr auto kSlices = record_slices<Payload>(ArgumentIndices<Payload>{});

// A payload without distances takes the medium form only, with its arguments whole, so a struct's payload is its bytes,
// which the host copies into its own struct as they are. Arguments go largest alignment first and, within one
// alignment, those whose bytes end on a multiple of it first, each at a multiple of its alignment up to 4, so none of
// its slices straddles a word.
template <typename Payload, size_t... I>
constexpr std::array<streaming_profiler::detail::ArgumentSlice, sizeof...(I)> place_arguments(
    std::index_sequence<I...>) {
    constexpr std::array<uint32_t, sizeof...(I)> alignments{uint32_t(alignof(ArgumentAt<Payload, I>))...};
    std::array<streaming_profiler::detail::ArgumentSlice, sizeof...(I)> placements{
        streaming_profiler::detail::ArgumentSlice{
            uint16_t(I), 0, 0, uint16_t(kFieldLayout<ArgumentAt<Payload, I>>.sent_bytes)}...};
    uint32_t end = 0;
    for (uint32_t align = std::max({uint32_t{1}, alignments[I]...}); align != 0; align /= 2) {
        for (const bool ragged : {false, true}) {
            for (size_t argument = 0; argument < placements.size(); ++argument) {
                if (alignments[argument] == align && (placements[argument].bytes % align != 0) == ragged) {
                    placements[argument].to = uint16_t(round_up(end, std::min<uint32_t>(align, 4)));
                    end = placements[argument].to + placements[argument].bytes;
                }
            }
        }
    }
    return placements;
}
template <typename Payload>
inline constexpr auto kArgumentPlacement = place_arguments<Payload>(ArgumentIndices<Payload>{});

// The words the record's time takes after word 0, and the bytes each Timestamp's distance takes, in each form.
constexpr uint32_t time_words(streaming_profiler::detail::TimeForm form) { return uint32_t(form); }
constexpr uint32_t distance_bytes(streaming_profiler::detail::TimeForm form) { return 2u << uint32_t(form); }

template <size_t N>
struct PlacedLayout {
    uint32_t body_words;
    std::array<streaming_profiler::detail::ArgumentSlice, N> slices;
};
template <typename Payload, streaming_profiler::detail::TimeForm F>
inline constexpr auto kLayout = [] {
    constexpr auto& slices = kSlices<Payload>;
    PlacedLayout<slices.size()> placed{};
    uint32_t end = 0;
    if constexpr (!kSendsDistances<Payload>) {
        for (size_t k = 0; k < slices.size(); ++k) {
            const uint32_t to = kArgumentPlacement<Payload>[slices[k].argument].to + slices[k].from;
            placed.slices[k] = {slices[k].argument, slices[k].from, uint16_t(to), slices[k].bytes};
            end = std::max(end, to + slices[k].bytes);
        }
    } else {
        // Each slice first-fits at a multiple of its size up to 4, widest first; a payload byte map with room for the
        // short form's time and every slice at its widest can't run out.
        bool taken[4 + 8 * slices.size()] = {};
        const auto is_free = [&](uint32_t offset, uint32_t bytes) {
            for (uint32_t byte = offset; byte < offset + bytes; ++byte) {
                if (taken[byte]) {
                    return false;
                }
            }
            return true;
        };
        if (F == streaming_profiler::detail::TimeForm::Short) {
            // The short form's record time takes bytes 2 and 3 of every call site's payload, so the decoder finds it
            // before it knows the call site.
            taken[2] = taken[3] = true;
            end = 4;
        }
        for (uint32_t align = 4; align != 0; align /= 2) {
            for (size_t k = 0; k < slices.size(); ++k) {
                const uint32_t bytes = slices[k].is_distance ? distance_bytes(F) : slices[k].bytes;
                if (std::min<uint32_t>(bytes, 4) != align) {
                    continue;
                }
                uint32_t offset = 0;
                while (!is_free(offset, bytes)) {
                    offset += align;
                }
                for (uint32_t byte = offset; byte < offset + bytes; ++byte) {
                    taken[byte] = true;
                }
                placed.slices[k] = {slices[k].argument, slices[k].from, uint16_t(offset), uint16_t(bytes)};
                end = std::max(end, offset + bytes);
            }
        }
    }
    placed.body_words = time_words(F) + (end + 3) / 4;
    return placed;
}();

// Each ELF has its own copy of the lane's cursor (internal/debug/device_publish_ring.h), and the host keeps
// one per lane. A kernel's launch resets its copy, so only kernels write short records.
#if defined(KERNEL_BUILD)
inline constexpr bool kWritesShortRecords = true;
#else
inline constexpr bool kWritesShortRecords = false;
#endif
template <typename Payload>
inline constexpr bool kHasShort = kWritesShortRecords && kSendsDistances<Payload> &&
                                  kLayout<Payload, streaming_profiler::detail::TimeForm::Short>.body_words <
                                      kLayout<Payload, streaming_profiler::detail::TimeForm::Medium>.body_words;
// The body length in word 0 of a call site's packets: its first form's, shortest first.
template <typename Payload>
inline constexpr uint32_t kFirstBody =
    kHasShort<Payload> ? kLayout<Payload, streaming_profiler::detail::TimeForm::Short>.body_words
                       : kLayout<Payload, streaming_profiler::detail::TimeForm::Medium>.body_words;
template <typename Payload, streaming_profiler::detail::TimeForm F>
inline constexpr uint32_t kPayloadWords = kLayout<Payload, F>.body_words - time_words(F);

// Each slice's first word among the words its record's slices take, an 8-byte slice taking two, low word first.
template <typename Payload>
inline constexpr auto kSliceFirstWord = [] {
    std::array<uint32_t, kSlices<Payload>.size() + 1> first{};
    for (size_t k = 0; k < kSlices<Payload>.size(); ++k) {
        first[k + 1] = first[k] + (kSlices<Payload>[k].bytes == 8 ? 2 : 1);
    }
    return first;
}();
// The words a record crosses an out-of-line writer's call in, as arguments in registers: its payload's when it has
// one form, and otherwise its slices', since the writer picks the form.
template <typename Payload>
struct RegisterWords {
    std::array<
        uint32_t,
        kSendsDistances<Payload> ? kSliceFirstWord<Payload>.back()
                                 : kPayloadWords<Payload, streaming_profiler::detail::TimeForm::Medium>>
        words;
};

// A record's payload, as the parts it stores in. A run is kRunWords or more payload words that are one argument's
// consecutive 4-aligned words as it lies in memory, which copy straight from there. Every other word is built from the
// slices that land in it. A shorter run builds its words too: the compiler keeps those in registers, and a copy would
// need the argument in memory.
inline constexpr uint32_t kRunWords = 9;
struct PayloadPart {
    uint16_t first_word = 0;
    uint16_t words = 1;
    int16_t argument = -1;  // the argument a run copies from; -1 for a built word
    uint16_t argument_offset = 0;
};
template <size_t kWords>
struct PayloadLayout {
    std::array<PayloadPart, kWords> parts{};
    size_t count = 0;
    bool has_run = false;
};

// Which argument and byte offset each payload word of form F copies whole from, or argument -1 for a word that must be
// built: one holding bytes of two arguments, a Timestamp's distance, or the short form's time.
template <typename Payload, streaming_profiler::detail::TimeForm F, size_t... I>
constexpr std::array<PayloadPart, kPayloadWords<Payload, F>> copyable_words(std::index_sequence<I...>) {
    constexpr auto& layout = kLayout<Payload, F>;
    constexpr auto& slices = kSlices<Payload>;
    constexpr uint32_t sizes[] = {uint32_t(sizeof(ArgumentAt<Payload, I>))...};
    constexpr uint32_t alignments[] = {uint32_t(alignof(ArgumentAt<Payload, I>))...};
    std::array<PayloadPart, kPayloadWords<Payload, F>> words{};
    for (uint32_t word = 0; word < words.size(); ++word) {
        // A copyable word's bytes all come from one argument, each at the same distance from its place in the payload.
        int32_t argument = -1, distance = 0;
        bool copyable = !(F == streaming_profiler::detail::TimeForm::Short && word == 0);
        for (uint32_t byte = 4 * word; byte < 4 * word + 4; ++byte) {
            for (size_t k = 0; k < slices.size(); ++k) {
                const streaming_profiler::detail::ArgumentSlice placed = layout.slices[k];
                if (placed.to <= byte && byte < placed.to + placed.bytes) {
                    const int32_t byte_distance = int32_t(placed.from + (byte - placed.to)) - int32_t(byte);
                    copyable &= !slices[k].is_distance &&
                                (argument < 0 || (argument == placed.argument && distance == byte_distance));
                    argument = placed.argument;
                    distance = byte_distance;
                }
            }
        }
        const int32_t offset = int32_t(4 * word) + distance;
        if (copyable && argument >= 0 && alignments[argument] >= 4 && offset >= 0 && offset % 4 == 0 &&
            uint32_t(offset) + 4 <= sizes[argument]) {
            words[word] = {uint16_t(word), 1, int16_t(argument), uint16_t(offset)};
        } else {
            words[word] = {uint16_t(word), 1};
        }
    }
    return words;
}
template <typename Payload, streaming_profiler::detail::TimeForm F, bool kCopyRuns>
inline constexpr auto kPayloadLayout = [] {
    constexpr auto words = copyable_words<Payload, F>(ArgumentIndices<Payload>{});
    PayloadLayout<words.size()> payload;
    for (uint32_t word = 0; word < words.size();) {
        const PayloadPart start = words[word];
        uint32_t end = word + 1;
        while (start.argument >= 0 && end < words.size() && words[end].argument == start.argument &&
               words[end].argument_offset == start.argument_offset + 4 * (end - word)) {
            ++end;
        }
        if (kCopyRuns && start.argument >= 0 && end - word >= kRunWords) {
            payload.parts[payload.count++] = {
                uint16_t(word), uint16_t(end - word), start.argument, start.argument_offset};
            payload.has_run = true;
            word = end;
        } else {
            payload.parts[payload.count++] = {uint16_t(word)};
            ++word;
        }
    }
    return payload;
}();
template <typename Payload>
inline constexpr bool kHasRun = kPayloadLayout<Payload, streaming_profiler::detail::TimeForm::Medium, true>.has_run;

template <typename>
struct MemberOf;
template <typename Class, typename Member>
struct MemberOf<Member Class::*> {
    using ClassType = Class;
    using MemberType = Member;
};
// The payload of a call with kPublishTimeIn<kField>: the struct kField belongs to, less the field kField names, which
// is the leaf its sample sets.
template <auto kField>
struct PublishTimePayload {
    typename MemberOf<decltype(kField)>::ClassType value;
};
template <auto kField>
constexpr size_t publish_time_leaf() {
    using Class = typename MemberOf<decltype(kField)>::ClassType;
    size_t leaf = 0;
    while (__builtin_bit_cast(uint64_t, sample_of<Class>(leaf).*kField) == 0) {
        ++leaf;
    }
    return leaf;
}
template <auto kField>
inline constexpr size_t kUnsentLeaf<PublishTimePayload<kField>> = publish_time_leaf<kField>();

}  // namespace kernel_profiler::records

// The writer stores records on the streaming profiler's rings, which kernels and firmware built with it have. Dispatch
// kernels have none: PROFILE_KERNEL is a global JIT define, so they get it too, but no relay serves a dispatch core, so
// a ring there would fill and wedge the next relay bring-up. Nor does its relay kernel: the ring's producer is ~1 KB
// the relay has no code room for (its self-profiling is its own staging-slot markers, see
// impl/streaming_profiler/kernels/drisc_relay.cpp).
#if defined(PROFILE_STREAMING) && defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && \
    !defined(STREAMING_PROFILER_RELAY_KERNEL)

#include "internal/debug/device_publish_ring.h"

namespace kernel_profiler::records {

template <size_t I, typename Source>
FORCE_INLINE const auto& argument_at(const Source& source) {
    if constexpr (streaming_profiler::detail::is_specialization<Source, std::tuple>) {
        return std::get<I>(source);
    } else {
        return source;
    }
}

// The record's arguments where the publishing call holds them: the payload struct it passed, or references to its
// arguments, so each payload word reads its argument where it lies rather than from a copy.
template <typename... Args>
FORCE_INLINE decltype(auto) record_source(const Args&... args) {
    if constexpr (streaming_profiler::detail::is_specialization<RecordPayload<Args...>, std::tuple>) {
        return std::tuple<const Args&...>(args...);
    } else {
        return (args, ...);
    }
}

template <uint32_t kBytes>
using UintOfSize = std::conditional_t<
    kBytes == 1,
    uint8_t,
    std::conditional_t<kBytes == 2, uint16_t, std::conditional_t<kBytes == 4, uint32_t, uint64_t>>>;

// Slice K's bits, from the record's arguments where the publishing call holds them or from its register words. The load
// is typed rather than a memcpy through an aligned pointer, so the fields of an argument held in registers stay there.
template <typename Payload, size_t K, typename Source>
FORCE_INLINE auto slice_bits(const Source& source) {
    constexpr Slice slice = kSlices<Payload>[K];
    using Bits = std::conditional_t<slice.bytes == 8, uint64_t, uint32_t>;
    if constexpr (std::is_same_v<Source, RegisterWords<Payload>>) {
        constexpr uint32_t first = kSliceFirstWord<Payload>[K];
        if constexpr (slice.bytes == 8) {
            return uint64_t{source.words[first + 1]} << 32 | source.words[first];
        } else {
            return Bits{source.words[first]};
        }
    } else {
        using Load [[gnu::may_alias]] = UintOfSize<slice.bytes>;
        const auto* bytes = reinterpret_cast<const unsigned char*>(&argument_at<slice.argument>(source)) + slice.from;
        return Bits{*reinterpret_cast<const Load*>(bytes)};
    }
}

// Slice K's bits where form F's payload word `word` holds them, or 0 when the slice lies elsewhere. Every caller passes
// a constant word, so the tests fold away.
template <typename Payload, streaming_profiler::detail::TimeForm F, size_t K, typename Bits>
FORCE_INLINE uint32_t slice_in_word(Bits bits, uint32_t word) {
    constexpr streaming_profiler::detail::ArgumentSlice copy = kLayout<Payload, F>.slices[K];
    if (copy.to / 4u == word) {
        return uint32_t(bits) << (8 * (copy.to % 4));
    }
    if (copy.bytes == 8 && copy.to / 4u + 1 == word) {
        return uint32_t(uint64_t(bits) >> 32);
    }
    return 0;
}

// Where a record's payload words go: ring slots after its head, or the ring word by word. Each takes a payload word's
// index, which only the slots use.
struct SlotStore {
    volatile tt_l1_ptr uint32_t* slots;

    FORCE_INLINE void write_word(uint32_t index, uint32_t value) const { slots[index] = value; }

    // Each batch's loads go ahead of its stores, so they overlap instead of each waiting out the one before: a copy a
    // word at a time runs slower than building the words in registers. Four words a batch copy a 32-word run in about
    // half the cycles of one at a time, with no more code; eight are about 10% faster again but larger.
    template <uint32_t kWords>
    FORCE_INLINE void copy_run(uint32_t index, const AliasingWord* from) const {
        constexpr uint32_t kBatch = 4;
        constexpr uint32_t kBatched = kWords / kBatch * kBatch;
        for (uint32_t done = 0; done < kBatched; done += kBatch) {
            copy_batch(slots + index + done, from + done, std::make_index_sequence<kBatch>{});
        }
        if constexpr (kWords != kBatched) {
            copy_batch(slots + index + kBatched, from + kBatched, std::make_index_sequence<kWords - kBatched>{});
        }
    }

private:
    template <size_t... B>
    static FORCE_INLINE void copy_batch(
        volatile tt_l1_ptr uint32_t* to, const AliasingWord* from, std::index_sequence<B...>) {
        const uint32_t batch[] = {from[B]...};
        ((to[B] = batch[B]), ...);
    }
};
struct WordStore {
    static FORCE_INLINE void write_word(uint32_t, uint32_t value) { ring_write_word(value); }
    template <uint32_t kWords>
    static FORCE_INLINE void copy_run(uint32_t, const AliasingWord* from) {
        ring_copy_words(from, kWords);
    }
};

// The writer of every record. Source is where it reads the record's arguments, as record_source gives them or as
// RegisterWords. A payload with Timestamps writes the shortest form they fit; any other payload is always medium.
template <typename Payload, typename Source>
struct RecordWriter {
    using Form = streaming_profiler::detail::TimeForm;

    // The longest packet: a full one, or a medium one with the timer's high word ahead of it.
    static constexpr uint32_t kMaxWords = [] {
        if constexpr (kSendsDistances<Payload>) {
            return 1 + kLayout<Payload, Form::Full>.body_words;
        } else {
            return 2 + kLayout<Payload, Form::Medium>.body_words;
        }
    }();

    // `inputs` are the record's register words, or the address of its source.
    template <typename... Inputs>
    static FORCE_INLINE void write(uint32_t header, uint32_t time_hi, uint32_t time_lo, Inputs... inputs) {
        if constexpr (kFromRegisters) {
            write_record(header, time_hi, time_lo, Source{{inputs...}}, kSliceIndices);
        } else {
            write_record(header, time_hi, time_lo, *inputs..., kSliceIndices);
        }
    }

private:
    static constexpr auto kSliceIndices = std::make_index_sequence<kSlices<Payload>.size()>{};
    static constexpr bool kFromRegisters = std::is_same_v<Source, RegisterWords<Payload>>;
    static constexpr bool kCopyRuns = !kFromRegisters;

    template <size_t K>
    static constexpr bool kIsDistance = kSlices<Payload>[K].is_distance;

    // The record being written: where its arguments are, its time, and each slice's Timestamp bits, which are read
    // before anything is written because they choose the form (0 for any other slice, which is read as it is stored).
    struct OutgoingRecord {
        const Source& source;
        uint32_t time_hi;
        uint32_t time_lo;
        std::array<uint64_t, kSlices<Payload>.size()> stamps;
    };

    template <size_t... K>
    static FORCE_INLINE void write_record(
        uint32_t header, uint32_t time_hi, uint32_t time_lo, const Source& source, std::index_sequence<K...>) {
        const OutgoingRecord record{source, time_hi, time_lo, {timestamp_bits<K>(source)...}};
        if constexpr (kSendsDistances<Payload>) {
            write_shortest_form(header, record, kSliceIndices);
        } else {
            write_form<Form::Medium>(header, record);
        }
    }
    template <size_t K>
    static FORCE_INLINE uint64_t timestamp_bits(const Source& source) {
        if constexpr (kIsDistance<K>) {
            return slice_bits<Payload, K>(source);
        } else {
            return 0;
        }
    }

    // A Timestamp's distance back from the record's time, its low word and its high word, which borrows when the
    // Timestamp is later, as a torn read is; 0 for any other slice.
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t distance_lo(const Record& record) {
        if constexpr (kIsDistance<K>) {
            return record.time_lo - uint32_t(record.stamps[K]);
        } else {
            return 0;
        }
    }
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t distance_hi(const Record& record) {
        if constexpr (kIsDistance<K>) {
            const uint64_t stamp = record.stamps[K];
            return record.time_hi - uint32_t(stamp >> 32) - (record.time_lo < uint32_t(stamp));
        } else {
            return 0;
        }
    }

    template <Form F, size_t K, typename Record>
    static FORCE_INLINE auto sent_slice(const Record& record) {
        if constexpr (kIsDistance<K> && F == Form::Full) {
            return uint64_t{distance_hi<K>(record)} << 32 | distance_lo<K>(record);
        } else if constexpr (kIsDistance<K>) {
            return distance_lo<K>(record);
        } else {
            return slice_bits<Payload, K>(record.source);
        }
    }

    // A short record needs its time within 16 bits after the lane's cursor and in the cursor's high-word period, and
    // every Timestamp within 16 bits back; a medium one needs every Timestamp within 32 bits back. No cursor (high word
    // kUnset) fails the high-word test. The tests are one OR-tree per form, and the short record, the commonest on a
    // busy lane, falls through first.
    template <typename Record, size_t... K>
    static FORCE_INLINE void write_shortest_form(uint32_t header, const Record& record, std::index_sequence<K...>) {
        const uint32_t beyond_32_bits = (distance_hi<K>(record) | ... | 0u);
        if constexpr (kHasShort<Payload>) {
            const uint32_t since_cursor = record.time_lo - g_cursor_lo;
            const uint32_t beyond_16_bits = (since_cursor | ... | distance_lo<K>(record)) >> 16;
            if (__builtin_expect((beyond_16_bits | (record.time_hi ^ g_cursor_hi) | beyond_32_bits) == 0, 1)) {
                write_form<Form::Short>(header, record);
                g_cursor_lo = record.time_lo;
                g_cursor_hi = record.time_hi;
                return;
            }
        }
        if (__builtin_expect(beyond_32_bits == 0, 1)) {
            write_form<Form::Medium>(header, record);
            if constexpr (kHasShort<Payload>) {
                g_cursor_lo = record.time_lo;
                g_cursor_hi = record.time_hi;
            }
            return;
        }
        write_full(header, record.time_hi, record.time_lo, sent_slice<Form::Full, K>(record)...);
    }

    // Out of line, so call sites don't carry the full form's code; its slices come as values.
    template <typename... Sent>
    [[gnu::noinline]] static void write_full(uint32_t header, uint32_t time_hi, uint32_t time_lo, Sent... sent) {
        ring_write_word(header + length_change<Form::Full>());
        ring_write_word(time_lo);
        ring_write_word(time_hi);
        write_full_payload(std::make_index_sequence<kPayloadWords<Payload, Form::Full>>{}, sent...);
    }
    template <size_t... W, typename... Sent>
    static FORCE_INLINE void write_full_payload(std::index_sequence<W...>, Sent... sent) {
        (ring_write_word(full_word<W>(std::index_sequence_for<Sent...>{}, sent...)), ...);
    }
    template <size_t W, size_t... K, typename... Sent>
    static FORCE_INLINE uint32_t full_word(std::index_sequence<K...>, Sent... sent) {
        return (slice_in_word<Payload, Form::Full, K>(sent, W) | ... | 0u);
    }

    // Word 0 is the call site's, with its first form's body length, so another form's adjusts it.
    template <Form F>
    static constexpr uint32_t length_change() {
        return (kLayout<Payload, F>.body_words - kFirstBody<Payload>) << ppfmt::LENGTH_SHIFT;
    }

    // A record stores as one block when it fits before the ring's end and is long enough for the check to pay for
    // itself: a medium record with payload, or one with a run. A short record is otherwise a word or two, cheaper word
    // by word.
    template <Form F, typename Record>
    static FORCE_INLINE void write_form(uint32_t header, const Record& record) {
        constexpr auto& payload = kPayloadLayout<Payload, F, kCopyRuns>;
        constexpr uint32_t kHeadWords = F == Form::Short ? 1 : 2;
        constexpr uint32_t kWords = kHeadWords + kPayloadWords<Payload, F>;
        constexpr bool kBlock = kWords > 2 && (F == Form::Medium || payload.has_run);
        constexpr auto parts = std::make_index_sequence<payload.count>{};
        // The short form keeps the time's distance from the cursor in the top half of its first payload word.
        const uint32_t short_time = F == Form::Short ? (record.time_lo - g_cursor_lo) << 16 : 0;
        if constexpr (F == Form::Medium) {
            ring_write_sticky_timer(record.time_hi);
        }
        if constexpr (kBlock) {
            const uint32_t index = wIndex;
            if (__builtin_expect(ring_fits_before_end(index, kWords), 1)) {
                volatile tt_l1_ptr uint32_t* const slots = ring_slots(index);
                slots[0] = header + length_change<F>();
                if constexpr (F == Form::Medium) {
                    slots[1] = record.time_lo;
                }
                store_payload<F>(SlotStore{slots + kHeadWords}, short_time, record, parts);
                wIndex = index + kWords;
                return;
            }
        }
        ring_write_word(header + length_change<F>());
        if constexpr (F == Form::Medium) {
            ring_write_word(record.time_lo);
        }
        store_payload<F>(WordStore{}, short_time, record, parts);
    }

    template <Form F, typename Store, typename Record, size_t... P>
    static FORCE_INLINE void store_payload(
        Store store, uint32_t short_time, const Record& record, std::index_sequence<P...>) {
        (store_part<F, kPayloadLayout<Payload, F, kCopyRuns>.parts[P]>(store, short_time, record), ...);
    }
    template <Form F, PayloadPart kPart, typename Store, typename Record>
    static FORCE_INLINE void store_part(Store store, uint32_t short_time, const Record& record) {
        if constexpr (kPart.argument < 0) {
            const uint32_t word = built_word<F, kPart.first_word>(record, kSliceIndices);
            store.write_word(kPart.first_word, kPart.first_word == 0 ? word | short_time : word);
        } else {
            const auto* argument = reinterpret_cast<const unsigned char*>(&argument_at<kPart.argument>(record.source));
            store.template copy_run<kPart.words>(
                kPart.first_word, reinterpret_cast<const AliasingWord*>(argument + kPart.argument_offset));
        }
    }

    template <Form F, size_t W, typename Record, size_t... K>
    static FORCE_INLINE uint32_t built_word(const Record& record, std::index_sequence<K...>) {
        if constexpr (kFromRegisters && !kSendsDistances<Payload>) {
            return record.source.words[W];
        } else {
            return (slice_in_word<Payload, F, K>(sent_slice<F, K>(record), W) | ... | 0u);
        }
    }
};

template <typename Payload, size_t A, typename Source, size_t... K>
FORCE_INLINE uint32_t register_word(const Source& source, std::index_sequence<K...>) {
    if constexpr (!kSendsDistances<Payload>) {
        return (
            slice_in_word<Payload, streaming_profiler::detail::TimeForm::Medium, K>(slice_bits<Payload, K>(source), A) |
            ... | 0u);
    } else {
        constexpr size_t slice = [] {
            size_t k = 0;
            while (kSliceFirstWord<Payload>[k + 1] <= A) {
                ++k;
            }
            return k;
        }();
        const auto bits = slice_bits<Payload, slice>(source);
        if constexpr (A == kSliceFirstWord<Payload>[slice]) {
            return uint32_t(bits);
        } else {
            return uint32_t(uint64_t(bits) >> 32);
        }
    }
}

// A record with a run crosses the writer's call by its source's address, which the run copies from; any other record
// as its register words, so a small record stays in registers when the writer isn't inlined.
template <bool kImmediately, bool kNoInline, typename Payload, typename Source, size_t... A>
FORCE_INLINE void publish_record(uint32_t header, const Source& source, std::index_sequence<A...>) {
    if constexpr (kHasRun<Payload>) {
        publish_with<RecordWriter<Payload, Source>, kImmediately, kNoInline>(header, &source);
    } else {
        constexpr auto slices = std::make_index_sequence<kSlices<Payload>.size()>{};
        publish_with<RecordWriter<Payload, RegisterWords<Payload>>, kImmediately, kNoInline>(
            header, register_word<Payload, A>(source, slices)...);
    }
}

}  // namespace kernel_profiler::records

namespace kernel_profiler {

static_assert(
    1 + records::kLayout<records::PublishTimePayload<&streaming_profiler::Interval::end>, streaming_profiler::detail::TimeForm::Full>.body_words ==
        MIN_RESERVE_WORDS,
    "every room check reserves a zone record's longest packet");

// Whether ppfmt's stall record, which write_pending_stall writes and the host describes stalls from, is the medium
// record DevicePublish lays out for a zone, whose Interval's Timestamps lie back to back.
constexpr bool stall_record_is_zone_medium_record() {
    using Payload = records::PublishTimePayload<&streaming_profiler::Interval::end>;
    constexpr auto layout = records::kLayout<Payload, streaming_profiler::detail::TimeForm::Medium>;
    return layout.body_words == ppfmt::STALL_BODY_WORDS && layout.slices.size() == 1 && layout.slices[0].to == 0 &&
           8u * layout.slices[0].bytes == ppfmt::STALL_START_DISTANCE_BITS &&
           records::kUnsentLeaf<Payload> == ppfmt::STALL_PUBLISH_TIME_FIELD &&
           offsetof(streaming_profiler::Interval, end) ==
               streaming_profiler::detail::code_bytes(streaming_profiler::detail::kTimestampCode);
}
static_assert(stall_record_is_zone_medium_record(), "the host describes stalls by ppfmt's stall record");

}  // namespace kernel_profiler

#endif
