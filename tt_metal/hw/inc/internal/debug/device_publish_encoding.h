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

// A struct's sample `sample`, which numbers its fields in binary (detail::sample_count), or with kEveryField, the
// struct with every field set.
inline constexpr size_t kEveryField = ~size_t{0};
struct SampleSource {
    size_t sample = 0;  // or kEveryField
    size_t fields_built = 0;
    constexpr void open() {}
    constexpr void close() {}
    // A float's or a double's smallest negative value sets its lowest and highest bits, a bool's true its only one.
    template <typename V>
    constexpr V leaf() {
        const size_t number_plus_one = ++fields_built;
        if (sample != kEveryField && ((number_plus_one >> sample) & 1) == 0) {
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
constexpr T sample_of(size_t sample) {
    SampleSource source{sample};
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
// padding bit zero, which nothing about how the compiler writes padding can change. Which of T's bits are value bits
// doesn't depend on the values, so the sample with every field set tells them.
template <typename T, size_t... K>
constexpr Bytes<sizeof(T) * sizeof...(K)> samples_of(std::index_sequence<K...>) {
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

// Where each of T's fields lies, which its descriptor carries and its field layout comes from.
template <typename T>
inline constexpr auto kFieldBits = [] {
    constexpr uint32_t kFields = uint32_t(detail::leaf_type_codes<T>.size());
    std::array<detail::FieldBits, kFields> fields{};
    if constexpr (kFields != 0) {
        constexpr uint32_t kSampleCount = detail::sample_count(kFields);
        const auto samples = samples_of<T>(std::make_index_sequence<kSampleCount>{});
        detail::field_bits_from_samples(samples.at, kSampleCount, uint32_t(sizeof(T)), fields.data());
    }
    return fields;
}();

constexpr uint32_t round_up(uint32_t bytes, uint32_t align) { return (bytes + align - 1) / align * align; }

// The leaf of an argument of type V that a record doesn't send (PublishTimeIn), or none.
template <typename V>
inline constexpr size_t kUnsentLeaf = ~size_t{0};

// Where each of V's fields lies, as its first byte and the bytes up to its last: a struct's from its field bits, so
// they are where the compiler put them whatever its bit-fields, packing or alignas, and an array's its element's,
// repeated. A record sends an argument's first sent_bytes, up to the end of the last field it sends.
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
            const detail::FieldBits bits = kFieldBits<V>[field];
            layout.offsets[field] = uint16_t(bits.bit / 8);
            layout.sizes[field] = uint16_t((bits.bit + bits.width + 7) / 8 - bits.bit / 8);
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
    bool is_delta;                   // a Timestamp the short and medium forms send as a distance back from the time
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
    // Each piece goes once, from the first field whose piece is the widest over the piece's first byte: aligned pieces
    // of 1, 2, 4 and 8 bytes either nest or don't meet, so a wider piece over that byte holds the whole piece.
    std::array<uint32_t, sizeof(V)> widest_at{};
    std::array<size_t, sizeof(V)> owner{};
    for (size_t field = 0; field < codes.size(); ++field) {
        for (uint32_t byte = piece_from[field]; byte < piece_from[field] + piece_bytes[field]; ++byte) {
            if (piece_bytes[field] > widest_at[byte]) {
                widest_at[byte] = piece_bytes[field];
                owner[byte] = field;
            }
        }
    }
    for (size_t field = 0; field < codes.size(); ++field) {
        if (piece_bytes[field] == 0 || owner[piece_from[field]] != field) {
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
constexpr bool sends_deltas(std::index_sequence<I...>) {
    return streaming_profiler::detail::holds<Payload>(streaming_profiler::detail::kTimestampCode) &&
           (kArgumentSlices<ArgumentAt<Payload, I>>.holds_every_field && ...);
}
template <typename Payload>
inline constexpr bool kSendsDeltas = sends_deltas<Payload>(ArgumentIndices<Payload>{});

// Puts argument `argument`'s slices in `slices`, from slices[k] on.
template <typename V>
constexpr void argument_slices(Slice* slices, size_t& k, uint16_t argument, bool as_deltas) {
    constexpr auto& own = kArgumentSlices<V>;
    for (size_t s = 0; s < own.count; ++s) {
        slices[k] = own.slices[s];
        slices[k].argument = argument;
        slices[k].is_delta = as_deltas && own.slices[s].is_delta;
        ++k;
    }
}
template <typename Payload, size_t... I>
constexpr auto record_slices(std::index_sequence<I...>) {
    std::array<Slice, (kArgumentSlices<ArgumentAt<Payload, I>>.count + ... + 0)> slices{};
    [[maybe_unused]] size_t count = 0;
    (argument_slices<ArgumentAt<Payload, I>>(slices.data(), count, uint16_t(I), kSendsDeltas<Payload>), ...);
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

// The words the record's time takes after word 0, and the bytes each Timestamp takes, in each form: its distance back
// from the record's time in the short and medium forms, and the Timestamp itself in the full form, where a distance
// would take the same 8 bytes.
constexpr uint32_t time_words(streaming_profiler::detail::TimestampWidth width) { return uint32_t(width); }
constexpr uint32_t timestamp_bytes(streaming_profiler::detail::TimestampWidth width) { return 2u << uint32_t(width); }

template <size_t N>
struct PlacedLayout {
    uint32_t packet_words;  // word 0 included
    std::array<streaming_profiler::detail::ArgumentSlice, N> slices;
};
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
inline constexpr auto kLayout = [] {
    constexpr auto& slices = kSlices<Payload>;
    PlacedLayout<slices.size()> placed{};
    uint32_t end = 0;
    if constexpr (!kSendsDeltas<Payload>) {
        for (size_t k = 0; k < slices.size(); ++k) {
            const uint32_t to = kArgumentPlacement<Payload>[slices[k].argument].to + slices[k].from;
            placed.slices[k] = {slices[k].argument, slices[k].from, uint16_t(to), slices[k].bytes};
            end = std::max(end, to + slices[k].bytes);
        }
    } else {
        // Each slice first-fits at a multiple of its size up to 4, widest first; a payload byte map with room for the
        // short form's time and every slice at its widest can't run out. Bytes only get taken, so a search for a size
        // starts where the last one for that size ended.
        bool taken[4 + 8 * slices.size()] = {};
        uint32_t search_from[9] = {};  // by size in bytes
        const auto is_free = [&](uint32_t offset, uint32_t bytes) {
            for (uint32_t byte = offset; byte < offset + bytes; ++byte) {
                if (taken[byte]) {
                    return false;
                }
            }
            return true;
        };
        if (kWidth == streaming_profiler::detail::TimestampWidth::Bits16) {
            // The short form's record time takes bytes 2 and 3 of every call site's payload, so the decoder finds it
            // before it knows the call site.
            taken[2] = taken[3] = true;
            end = 4;
        }
        for (uint32_t align = 4; align != 0; align /= 2) {
            for (size_t k = 0; k < slices.size(); ++k) {
                const uint32_t bytes = slices[k].is_delta ? timestamp_bytes(kWidth) : slices[k].bytes;
                if (std::min<uint32_t>(bytes, 4) != align) {
                    continue;
                }
                uint32_t offset = search_from[bytes];
                while (!is_free(offset, bytes)) {
                    offset += align;
                }
                search_from[bytes] = offset;
                for (uint32_t byte = offset; byte < offset + bytes; ++byte) {
                    taken[byte] = true;
                }
                placed.slices[k] = {slices[k].argument, slices[k].from, uint16_t(offset), uint16_t(bytes)};
                end = std::max(end, offset + bytes);
            }
        }
    }
    placed.packet_words = 1 + time_words(kWidth) + (end + 3) / 4;
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
inline constexpr bool kHasBits16 =
    kWritesShortRecords && kSendsDeltas<Payload> &&
    kLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits16>.packet_words <
        kLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits32>.packet_words;
// The body length in word 0 of a call site's packets: its first form's, shortest first.
template <typename Payload>
inline constexpr uint32_t kFirstPacketWords =
    kHasBits16<Payload> ? kLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits16>.packet_words
                        : kLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits32>.packet_words;
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
inline constexpr uint32_t kPayloadWords = kLayout<Payload, kWidth>.packet_words - 1 - time_words(kWidth);

// A slice's bits in one payload word: the slice, the bit its bits start at in the word, and whether the word holds
// the high half of an 8-byte slice.
struct WordPiece {
    uint16_t slice = 0;
    uint8_t shift = 0;
    bool high_half = false;
};
// A payload word's pieces, lowest slice first: at most four, since a slice of up to 4 bytes lies in one word and an
// 8-byte one fills the second of its two.
struct WordPieces {
    std::array<WordPiece, 4> pieces{};
    uint32_t count = 0;
};
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
inline constexpr auto kWordPieces = [] {
    constexpr auto& slices = kLayout<Payload, kWidth>.slices;
    std::array<WordPieces, kPayloadWords<Payload, kWidth>> words{};
    for (size_t k = 0; k < slices.size(); ++k) {
        WordPieces& first = words[slices[k].to / 4u];
        first.pieces[first.count++] = {uint16_t(k), uint8_t(8 * (slices[k].to % 4)), false};
        if (slices[k].bytes == 8) {
            WordPieces& second = words[slices[k].to / 4u + 1];
            second.pieces[second.count++] = {uint16_t(k), 0, true};
        }
    }
    return words;
}();
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth, size_t W>
using WordPieceIndices = std::make_index_sequence<kWordPieces<Payload, kWidth>[W].count>;

// Each slice's first word among the words its record's slices take, an 8-byte slice taking two, low word first.
template <typename Payload>
inline constexpr auto kSliceFirstWord = [] {
    std::array<uint32_t, kSlices<Payload>.size() + 1> first{};
    for (size_t k = 0; k < kSlices<Payload>.size(); ++k) {
        first[k + 1] = first[k] + (kSlices<Payload>[k].bytes == 8 ? 2 : 1);
    }
    return first;
}();
// The slice each of those words holds.
template <typename Payload>
inline constexpr auto kRegisterWordSlice = [] {
    std::array<uint16_t, kSliceFirstWord<Payload>.back()> slices{};
    for (size_t k = 0; k < kSlices<Payload>.size(); ++k) {
        for (uint32_t word = kSliceFirstWord<Payload>[k]; word < kSliceFirstWord<Payload>[k + 1]; ++word) {
            slices[word] = uint16_t(k);
        }
    }
    return slices;
}();
// The slices a record sends as distances back from its time.
template <typename Payload>
inline constexpr auto kDeltaSlices = [] {
    constexpr size_t kCount = [] {
        size_t count = 0;
        for (const Slice& slice : kSlices<Payload>) {
            count += slice.is_delta ? 1 : 0;
        }
        return count;
    }();
    std::array<uint16_t, kCount> deltas{};
    size_t count = 0;
    for (size_t k = 0; k < kSlices<Payload>.size(); ++k) {
        if (kSlices<Payload>[k].is_delta) {
            deltas[count++] = uint16_t(k);
        }
    }
    return deltas;
}();

// The words a record crosses an out-of-line writer's call in, as arguments in registers: its payload's when it has
// one form, and otherwise its slices', since the writer picks the form.
template <typename Payload>
struct RegisterWords {
    std::array<
        uint32_t,
        kSendsDeltas<Payload> ? kSliceFirstWord<Payload>.back()
                              : kPayloadWords<Payload, streaming_profiler::detail::TimestampWidth::Bits32>>
        words;
};

// A record's payload, as the parts it stores in. A run is kRunWords or more payload words that are one argument's
// consecutive 4-aligned words as it lies in memory, which copy straight from there. A loop is kLoopWords or more words
// that repeat a group of up to kMaxLoopPeriod words, each piece of each a constant stride further into its argument
// than the same piece a group before, which a loop builds from where the argument lies. Every other word is built from
// the slices that land in it. A shorter run builds its words too: the compiler keeps those in registers, and a copy
// would need the argument in memory. Built words run faster straight-line until there are so many that fetching their
// code holds them up: on a Tensix BRISC at O2, 32 halfword-pair words took 238 cycles straight-line and 249 in a loop,
// 48 took 463 and 364, and 256 took 5325 and 1830.
inline constexpr uint32_t kRunWords = 9;
inline constexpr uint32_t kLoopWords = 40;
inline constexpr uint32_t kMaxLoopPeriod = 4;
// One piece of a word a loop builds: where it loads from in its argument the first time, how much further on each
// next time, how many bytes it loads, the bit it goes at in the word, and whether it is a Timestamp's distance back
// from the record's time, which loads the Timestamp's low word. An 8-byte slice's piece loads the half it sends.
struct LoopPiece {
    uint16_t argument = 0;
    uint16_t from = 0;
    int16_t stride = 0;
    uint8_t bytes = 0;
    uint8_t shift = 0;
    bool distance = false;
};
// A payload word's pieces as a loop loads them, lowest slice first.
struct LoopWord {
    std::array<LoopPiece, 4> pieces{};
    uint32_t count = 0;
};
// `repeats` groups of `period` words, the first from the payload part's first word on.
struct PayloadLoop {
    uint32_t period = 0;
    uint32_t repeats = 0;
    std::array<LoopWord, kMaxLoopPeriod> words{};
};
struct PayloadPart {
    uint16_t first_word = 0;
    uint16_t words = 1;
    int16_t argument = -1;  // the argument a run copies from, or -1
    uint16_t argument_offset = 0;
    int16_t loop = -1;  // the loop that builds the part, or -1
};
template <size_t kWords>
struct PayloadLayout {
    std::array<PayloadPart, kWords> parts{};
    size_t count = 0;
    std::array<PayloadLoop, kWords / kLoopWords + 1> loops{};
    size_t loop_count = 0;
    bool has_run = false;
};

// Which argument and byte offset each payload word of form kWidth copies whole from, or argument -1 for a word that
// must be built: one holding bytes of two arguments, a Timestamp's distance (short and medium forms), or the short
// form's time.
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth, size_t... I>
constexpr std::array<PayloadPart, kPayloadWords<Payload, kWidth>> copyable_words(std::index_sequence<I...>) {
    constexpr auto& layout = kLayout<Payload, kWidth>;
    constexpr auto& slices = kSlices<Payload>;
    constexpr uint32_t sizes[] = {uint32_t(sizeof(ArgumentAt<Payload, I>))...};
    constexpr uint32_t alignments[] = {uint32_t(alignof(ArgumentAt<Payload, I>))...};
    std::array<PayloadPart, kPayloadWords<Payload, kWidth>> words{};
    for (uint32_t word = 0; word < words.size(); ++word) {
        // A copyable word's bytes all come from one argument, each at the same distance from its place in the payload.
        int32_t argument = -1, distance = 0;
        bool copyable = !(kWidth == streaming_profiler::detail::TimestampWidth::Bits16 && word == 0);
        const WordPieces& pieces = kWordPieces<Payload, kWidth>[word];
        for (uint32_t piece = 0; piece < pieces.count; ++piece) {
            const uint16_t k = pieces.pieces[piece].slice;
            const streaming_profiler::detail::ArgumentSlice placed = layout.slices[k];
            const int32_t slice_distance = int32_t(placed.from) - int32_t(placed.to);
            copyable &= !(slices[k].is_delta && kWidth != streaming_profiler::detail::TimestampWidth::Bits64) &&
                        (argument < 0 || (argument == placed.argument && distance == slice_distance));
            argument = placed.argument;
            distance = slice_distance;
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
// How many pieces form kWidth's payload word `word` has for a loop to load: none in the short form's first word, which
// holds the time's distance from the cursor.
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
constexpr uint32_t loop_piece_count(uint32_t word) {
    return kWidth == streaming_profiler::detail::TimestampWidth::Bits16 && word == 0
               ? 0
               : kWordPieces<Payload, kWidth>[word].count;
}
// Piece `piece` of form kWidth's payload word `word` as a loop loads it, stride 0.
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
constexpr LoopPiece loop_piece_at(uint32_t word, uint32_t piece) {
    const WordPiece& at = kWordPieces<Payload, kWidth>[word].pieces[piece];
    const Slice& slice = kSlices<Payload>[at.slice];
    const bool distance = slice.is_delta && kWidth != streaming_profiler::detail::TimestampWidth::Bits64;
    return {
        slice.argument,
        uint16_t(slice.from + (at.high_half ? 4 : 0)),
        0,
        uint8_t(distance || slice.bytes == 8 ? 4 : slice.bytes),
        at.shift,
        distance};
}
// How many times the group of `period` payload words from `word` on repeats from there: the same pieces each time,
// each a constant stride further on than the time before.
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
constexpr uint32_t loop_repeats(uint32_t word, uint32_t period) {
    uint32_t repeats = 1;
    for (; word + (repeats + 1) * period <= kPayloadWords<Payload, kWidth>; ++repeats) {
        for (uint32_t w = word; w < word + period; ++w) {
            const uint32_t count = loop_piece_count<Payload, kWidth>(w);
            if (count == 0 || loop_piece_count<Payload, kWidth>(w + repeats * period) != count) {
                return repeats;
            }
            for (uint32_t piece = 0; piece < count; ++piece) {
                const LoopPiece a = loop_piece_at<Payload, kWidth>(w, piece);
                const LoopPiece c = loop_piece_at<Payload, kWidth>(w + repeats * period, piece);
                const int32_t stride = int32_t(loop_piece_at<Payload, kWidth>(w + period, piece).from) - a.from;
                if (a.argument != c.argument || a.bytes != c.bytes || a.shift != c.shift || a.distance != c.distance ||
                    stride != int16_t(stride) || int32_t(c.from) - a.from != stride * int32_t(repeats)) {
                    return repeats;
                }
            }
        }
    }
    return repeats;
}
// The loop that builds the most payload words from `word` on, at the shortest period among those that build as many,
// when it builds kLoopWords or more; otherwise none.
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth>
constexpr PayloadLoop loop_from(uint32_t word) {
    PayloadLoop loop;
    for (uint32_t period = 1; period <= kMaxLoopPeriod && word + kLoopWords <= kPayloadWords<Payload, kWidth>;
         ++period) {
        const uint32_t repeats = loop_repeats<Payload, kWidth>(word, period);
        if (period * repeats >= kLoopWords && period * repeats > loop.period * loop.repeats) {
            loop.period = period;
            loop.repeats = repeats;
        }
    }
    for (uint32_t w = 0; w < loop.period; ++w) {
        LoopWord& pieces = loop.words[w];
        for (; pieces.count < loop_piece_count<Payload, kWidth>(word + w); ++pieces.count) {
            LoopPiece& piece = pieces.pieces[pieces.count];
            piece = loop_piece_at<Payload, kWidth>(word + w, pieces.count);
            piece.stride =
                int16_t(loop_piece_at<Payload, kWidth>(word + loop.period + w, pieces.count).from - piece.from);
        }
    }
    return loop;
}
template <typename Payload, streaming_profiler::detail::TimestampWidth kWidth, bool kCopyRuns>
inline constexpr auto kPayloadLayout = [] {
    constexpr auto words = copyable_words<Payload, kWidth>(ArgumentIndices<Payload>{});
    PayloadLayout<words.size()> payload;
    for (uint32_t word = 0; word < words.size();) {
        const PayloadPart start = words[word];
        uint32_t end = word + 1;
        while (kCopyRuns && start.argument >= 0 && end < words.size() && words[end].argument == start.argument &&
               words[end].argument_offset == start.argument_offset + 4 * (end - word)) {
            ++end;
        }
        if (kCopyRuns && start.argument >= 0 && end - word >= kRunWords) {
            payload.parts[payload.count++] = {
                uint16_t(word), uint16_t(end - word), start.argument, start.argument_offset};
            payload.has_run = true;
            word = end;
            continue;
        }
        const PayloadLoop loop = kCopyRuns ? loop_from<Payload, kWidth>(word) : PayloadLoop{};
        if (loop.period != 0) {
            payload.parts[payload.count++] = {
                uint16_t(word), uint16_t(loop.period * loop.repeats), -1, 0, int16_t(payload.loop_count)};
            payload.loops[payload.loop_count++] = loop;
            word += loop.period * loop.repeats;
        } else {
            payload.parts[payload.count++] = {uint16_t(word)};
            ++word;
        }
    }
    return payload;
}();
// Whether a record's writer reads its arguments where the call holds them, which a run's copy and a loop need: any of
// its forms has one.
template <typename Payload>
inline constexpr bool kReadsArgumentsInPlace =
    kPayloadLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits32, true>.has_run ||
    kPayloadLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits32, true>.loop_count != 0 ||
    (kSendsDeltas<Payload> &&
     kPayloadLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits16, true>.loop_count != 0) ||
    (kSendsDeltas<Payload> &&
     kPayloadLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits64, true>.loop_count != 0);

template <typename>
struct MemberOf;
template <typename Class, typename Member>
struct MemberOf<Member Class::*> {
    using ClassType = Class;
    using MemberType = Member;
};
// The payload of a call with kPublishTimeIn<kField>: the struct kField belongs to, less the field kField names, which
// is the leaf whose number the samples spell in kField's bits.
template <auto kField>
struct PublishTimePayload {
    typename MemberOf<decltype(kField)>::ClassType value;
};
template <auto kField>
constexpr size_t publish_time_leaf() {
    using Class = typename MemberOf<decltype(kField)>::ClassType;
    size_t number_plus_one = 0;
    for (uint32_t sample = 0; sample < detail::sample_count(uint32_t(detail::leaf_type_codes<Class>.size()));
         ++sample) {
        if (__builtin_bit_cast(uint64_t, sample_of<Class>(sample).*kField) != 0) {
            number_plus_one |= size_t{1} << sample;
        }
    }
    return number_plus_one - 1;
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
        const unsigned char* bytes;
        if constexpr (detail::is_specialization<Source, std::tuple>) {
            bytes = reinterpret_cast<const unsigned char*>(&std::get<slice.argument>(source)) + slice.from;
        } else {
            bytes = reinterpret_cast<const unsigned char*>(&source) + slice.from;
        }
        return Bits{*reinterpret_cast<const Load*>(bytes)};
    }
}

// Where a record's words go, each by its index in the record: slots from its first word's, when it fits before the
// ring's end, or ring indexes from its first word's, which wrap (WrapStore, device_publish_ring.h).
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

// The writer of every record. Source is where it reads the record's arguments, as record_source gives them or as
// RegisterWords. A payload with Timestamps writes the shortest form they fit; any other payload is always medium.
template <typename Payload, typename Source>
struct RecordWriter {
    using Width = streaming_profiler::detail::TimestampWidth;

    // The longest packet: a full one, or a medium one with the timer's high word ahead of it.
    static constexpr uint32_t kMaxWords = kSendsDeltas<Payload> ? kLayout<Payload, Width::Bits64>.packet_words
                                                                : 1 + kLayout<Payload, Width::Bits32>.packet_words;

    // `inputs` are the record's register words, or the address of its source. Returns the write index past the record.
    template <typename... Inputs>
    static FORCE_INLINE uint32_t write(uint32_t header, uint32_t time_hi, uint32_t time_lo, Inputs... inputs) {
        if constexpr (kFromRegisters) {
            return write_record(header, time_hi, time_lo, Source{{inputs...}}, kSliceIndices);
        } else {
            return write_record(header, time_hi, time_lo, *inputs..., kSliceIndices);
        }
    }

private:
    static constexpr auto kSliceIndices = std::make_index_sequence<kSlices<Payload>.size()>{};
    static constexpr auto kDeltaIndices = std::make_index_sequence<kDeltaSlices<Payload>.size()>{};
    static constexpr bool kFromRegisters = std::is_same_v<Source, RegisterWords<Payload>>;
    static constexpr bool kCopyRuns = !kFromRegisters;

    template <size_t K>
    static constexpr bool kIsDelta = kSlices<Payload>[K].is_delta;

    // The record being written: where its arguments are, its time, and each slice's Timestamp bits, which are read
    // before anything is written because they choose the form (0 for any other slice, which is read as it is stored).
    struct OutgoingRecord {
        const Source& source;
        uint32_t time_hi;
        uint32_t time_lo;
        std::array<uint64_t, kSlices<Payload>.size()> stamps;
    };

    template <size_t... K>
    static FORCE_INLINE uint32_t
    write_record(uint32_t header, uint32_t time_hi, uint32_t time_lo, const Source& source, std::index_sequence<K...>) {
        const OutgoingRecord record{source, time_hi, time_lo, {timestamp_bits<K>(source)...}};
        if constexpr (kSendsDeltas<Payload>) {
            return write_narrowest(header, record, kDeltaIndices);
        } else {
            return write_at_width<Width::Bits32>(header, record);
        }
    }
    template <size_t K>
    static FORCE_INLINE uint64_t timestamp_bits(const Source& source) {
        if constexpr (kIsDelta<K>) {
            return slice_bits<Payload, K>(source);
        } else {
            return 0;
        }
    }

    // Timestamp K's distance back from the record's time, its low word and its high word, which borrows when the
    // Timestamp is later, as a torn read is.
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t delta_lo(const Record& record) {
        return record.time_lo - uint32_t(record.stamps[K]);
    }
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t delta_hi(const Record& record) {
        const uint64_t stamp = record.stamps[K];
        return record.time_hi - uint32_t(stamp >> 32) - (record.time_lo < uint32_t(stamp));
    }
    // Nonzero when Timestamp K's high half differs from the record time's.
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t other_high_half(const Record& record) {
        return record.time_hi ^ uint32_t(record.stamps[K] >> 32);
    }

    // Writes the record in the narrowest width it fits. Every record of a call site with a 16-bit width moves the
    // lane's cursor to its publish Timestamp, whichever width it took. A 16-bit record needs its publish Timestamp
    // within 16 bits after the cursor and every payload Timestamp within 16 bits before it, all in the cursor's high
    // half: the high halves are compared rather than the deltas' high halves computed, which sends the rare record that
    // straddles a low-half wrap at 32 bits but keeps the 64-bit subtraction off the commonest path. A 32-bit record
    // needs every payload Timestamp within 32 bits before it. No cursor (high half kUnset) fails the 16-bit test.
    template <typename Record, size_t... D>
    static FORCE_INLINE uint32_t write_narrowest(uint32_t header, const Record& record, std::index_sequence<D...>) {
        if constexpr (kHasBits16<Payload>) {
            const uint32_t since_cursor = record.time_lo - g_cursor_lo;
            const uint32_t beyond_16_bits = (since_cursor | ... | delta_lo<kDeltaSlices<Payload>[D]>(record)) >> 16;
            const uint32_t other_high_halves =
                ((record.time_hi ^ g_cursor_hi) | ... | other_high_half<kDeltaSlices<Payload>[D]>(record));
            if (__builtin_expect((beyond_16_bits | other_high_halves) == 0, 1)) {
                const uint32_t end = write_at_width<Width::Bits16>(header, record);
                move_cursor(record.time_hi, record.time_lo);
                return end;
            }
        }
        if (__builtin_expect((delta_hi<kDeltaSlices<Payload>[D]>(record) | ... | 0u) == 0, 1)) {
            const uint32_t end = write_at_width<Width::Bits32>(header, record);
            move_cursor(record.time_hi, record.time_lo);
            return end;
        }
        // The 64-bit writer moves the cursor itself, so the inline code keeps no time across its call.
        if constexpr (kCopyRuns) {
            return write_bits64(header, record.time_hi, record.time_lo, &record.source);
        } else {
            return write_bits64_sent(header, record, kSliceIndices);
        }
    }
    template <typename Record, size_t... K>
    static FORCE_INLINE uint32_t write_bits64_sent(uint32_t header, const Record& record, std::index_sequence<K...>) {
        return write_bits64(header, record.time_hi, record.time_lo, slice_bits<Payload, K>(record.source)...);
    }
    static FORCE_INLINE void move_cursor(uint32_t time_hi, uint32_t time_lo) {
        if constexpr (kHasBits16<Payload>) {
            g_cursor_lo = time_lo;
            g_cursor_hi = time_hi;
        }
    }

    // Out of line, so call sites don't carry the full form's code. As at the writer's own call, a record with a run
    // comes as its source's address, and any other as values: its slices as the full form sends them.
    template <typename... Sent>
    [[gnu::noinline]] static uint32_t write_bits64(uint32_t header, uint32_t time_hi, uint32_t time_lo, Sent... sent) {
        const WrapStore store{wIndex};
        if constexpr (kCopyRuns) {
            store_full_record(store, header, time_hi, time_lo, *sent..., kSliceIndices);
        } else {
            store.write_word(0, header + length_change<Width::Bits64>());
            store.write_word(1, time_lo);
            store.write_word(2, time_hi);
            write_full_payload(store, std::make_index_sequence<kPayloadWords<Payload, Width::Bits64>>{}, sent...);
        }
        const uint32_t end = store.index + kWords<Width::Bits64>;
        wIndex = end;
        move_cursor(time_hi, time_lo);
        return end;
    }
    template <size_t... K>
    static FORCE_INLINE void store_full_record(
        WrapStore store,
        uint32_t header,
        uint32_t time_hi,
        uint32_t time_lo,
        const Source& source,
        std::index_sequence<K...>) {
        store_record<Width::Bits64>(
            store, header, 0, OutgoingRecord{source, time_hi, time_lo, {timestamp_bits<K>(source)...}});
    }
    template <size_t... W, typename... Sent>
    static FORCE_INLINE void write_full_payload(WrapStore store, std::index_sequence<W...>, Sent... sent) {
        const uint64_t values[] = {uint64_t{sent}...};
        (store.write_word(3 + W, full_word<W, Sent...>(values, WordPieceIndices<Payload, Width::Bits64, W>{})), ...);
    }
    // The full form's payload word W, from its slices as the out-of-line writer takes them, each widened to 64 bits.
    template <size_t W, typename... Sent, size_t... J>
    static FORCE_INLINE uint32_t full_word(const uint64_t* values, std::index_sequence<J...>) {
        constexpr const WordPieces& kWord = kWordPieces<Payload, Width::Bits64>[W];
        return (
            (uint32_t(
                 uint64_t(__type_pack_element<kWord.pieces[J].slice, Sent...>(values[kWord.pieces[J].slice])) >>
                 (kWord.pieces[J].high_half ? 32 : 0))
             << kWord.pieces[J].shift) |
            ... | 0u);
    }

    // Word 0 is the call site's, with its first form's body length, so another form's adjusts it.
    template <Width kWidth>
    static constexpr uint32_t length_change() {
        return (kLayout<Payload, kWidth>.packet_words - kFirstPacketWords<Payload>) << ppfmt::PACKET_WORDS_SHIFT;
    }

    // A record's words ahead of its payload: word 0, then the time's low word (medium and full) and high word (full).
    template <Width kWidth>
    static constexpr uint32_t kHeadWords = kWidth == Width::Bits16 ? 1 : (kWidth == Width::Bits32 ? 2 : 3);
    template <Width kWidth>
    static constexpr uint32_t kWords = kHeadWords<kWidth> + kPayloadWords<Payload, kWidth>;

    // A record stores as one block when it fits before the ring's end and is long enough for the check to pay for
    // itself: a medium record with payload, or one with a run. Otherwise, a short record of a word or two or one that
    // doesn't fit, each word goes to its own ring index, which wraps. Either way the write index stays in a register
    // and wIndex is stored once. Returns the write index past the record.
    template <Width kWidth, typename Record>
    static FORCE_INLINE uint32_t write_at_width(uint32_t header, const Record& record) {
        constexpr bool kBlock = kWords<kWidth> > 2 && (kWidth != Width::Bits16 || kPayload<kWidth>.has_run ||
                                                       kPayload<kWidth>.loop_count != 0);
        // The short form keeps the time's distance from the cursor in the top half of its first payload word.
        const uint32_t cursor_delta = kWidth == Width::Bits16 ? (record.time_lo - g_cursor_lo) << 16 : 0;
        if constexpr (kWidth == Width::Bits32) {
            ring_write_sticky_timer(record.time_hi);
        }
        const uint32_t index = wIndex;
        const uint32_t end = index + kWords<kWidth>;
        if constexpr (kBlock) {
            if (__builtin_expect(ring_fits_before_end(index, kWords<kWidth>), 1)) {
                store_record<kWidth>(SlotStore{ring_slots(index)}, header, cursor_delta, record);
                wIndex = end;
                return end;
            }
        }
        store_record<kWidth>(WrapStore{index}, header, cursor_delta, record);
        wIndex = end;
        return end;
    }

    template <Width kWidth, typename Store, typename Record>
    static FORCE_INLINE void store_record(Store store, uint32_t header, uint32_t cursor_delta, const Record& record) {
        store.write_word(0, header + length_change<kWidth>());
        if constexpr (kWidth != Width::Bits16) {
            store.write_word(1, record.time_lo);
        }
        if constexpr (kWidth == Width::Bits64) {
            store.write_word(2, record.time_hi);
        }
        store_payload<kWidth>(store, cursor_delta, record, std::make_index_sequence<kPayload<kWidth>.count>{});
    }
    template <Width kWidth, typename Store, typename Record, size_t... P>
    static FORCE_INLINE void store_payload(
        Store store, uint32_t cursor_delta, const Record& record, std::index_sequence<P...>) {
        (store_part<kWidth, P>(store, cursor_delta, record), ...);
    }
    template <Width kWidth>
    static constexpr const auto& kPayload = kPayloadLayout<Payload, kWidth, kCopyRuns>;
    template <Width kWidth, size_t P, typename Store, typename Record>
    static FORCE_INLINE void store_part(Store store, uint32_t cursor_delta, const Record& record) {
        constexpr PayloadPart kPart = kPayload<kWidth>.parts[P];
        constexpr uint32_t kWord = kHeadWords<kWidth> + kPart.first_word;
        if constexpr (kPart.loop >= 0) {
            store_loop<kWidth, kPart.loop>(store, kWord, record);
        } else if constexpr (kPart.argument < 0) {
            const uint32_t word = built_word<kWidth, kPart.first_word>(record);
            store.write_word(kWord, kPart.first_word == 0 ? word | cursor_delta : word);
        } else {
            const auto* argument = reinterpret_cast<const unsigned char*>(&argument_at<kPart.argument>(record.source));
            store.template copy_run<kPart.words>(
                kWord, reinterpret_cast<const AliasingWord*>(argument + kPart.argument_offset));
        }
    }

    // Loop L's words from the record's word `word` on, about kLoopBatch a batch in whole groups, each batch built, its
    // loads issued, before any of it is stored, so an L1 load's 8 cycles pass under the batch's other loads. On a
    // Tensix BRISC at O2, 16 words a batch built 256 words in 1830 cycles, against 1967 to 2453 for 4 or 8 a batch,
    // with or without each batch's loads going ahead of the stores of the one before.
    static constexpr uint32_t kLoopBatch = 16;
    template <Width kWidth, size_t L, typename Store, typename Record>
    static FORCE_INLINE void store_loop(Store store, uint32_t word, const Record& record) {
        constexpr PayloadLoop kLoop = kPayload<kWidth>.loops[L];
        constexpr uint32_t kGroups = std::max<uint32_t>(1, kLoopBatch / kLoop.period);  // a batch's
        constexpr uint32_t kBatches = kLoop.repeats / kGroups;
        constexpr uint32_t kRest = kLoop.repeats % kGroups;
        for (uint32_t batch = 0; batch < kBatches; ++batch) {
            store_batch<kWidth, L>(
                store,
                word + batch * kGroups * kLoop.period,
                record,
                batch * kGroups,
                std::make_index_sequence<kGroups * kLoop.period>{});
        }
        if constexpr (kRest != 0) {
            store_batch<kWidth, L>(
                store,
                word + kBatches * kGroups * kLoop.period,
                record,
                kBatches * kGroups,
                std::make_index_sequence<kRest * kLoop.period>{});
        }
    }
    // Loop L's words from `word` on, from its group `repeat` on.
    template <Width kWidth, size_t L, typename Store, typename Record, size_t... B>
    static FORCE_INLINE void store_batch(
        Store store, uint32_t word, const Record& record, uint32_t repeat, std::index_sequence<B...>) {
        constexpr const PayloadLoop& kLoop = kPayload<kWidth>.loops[L];
        const uint32_t built[] = {loop_word<kWidth, L, B % kLoop.period>(
            record, repeat + B / kLoop.period, std::make_index_sequence<kLoop.words[B % kLoop.period].count>{})...};
        ((store.write_word(word + B, built[B])), ...);
    }
    // Word W of loop L's group `repeat`: each piece's bits, loaded `repeat` strides on from the first group's.
    template <Width kWidth, size_t L, size_t W, typename Record, size_t... P>
    static FORCE_INLINE uint32_t loop_word(const Record& record, uint32_t repeat, std::index_sequence<P...>) {
        constexpr const LoopWord& kWord = kPayload<kWidth>.loops[L].words[W];
        return ((loop_piece<kWord.pieces[P]>(record, repeat) << kWord.pieces[P].shift) | ... | 0u);
    }
    template <LoopPiece kPiece, typename Record>
    static FORCE_INLINE uint32_t loop_piece(const Record& record, uint32_t repeat) {
        using Load [[gnu::may_alias]] = UintOfSize<kPiece.bytes>;
        const auto* argument = reinterpret_cast<const unsigned char*>(&argument_at<kPiece.argument>(record.source));
        const uint32_t bits = *reinterpret_cast<const Load*>(argument + kPiece.from + int32_t(repeat) * kPiece.stride);
        return kPiece.distance ? record.time_lo - bits : bits;
    }

    template <Width kWidth, size_t W, typename Record>
    static FORCE_INLINE uint32_t built_word(const Record& record) {
        if constexpr (kFromRegisters && !kSendsDeltas<Payload>) {
            return record.source.words[W];
        } else {
            return built_word<kWidth, W>(record, WordPieceIndices<Payload, kWidth, W>{});
        }
    }
    // Form kWidth's payload word W: each of its pieces where the word holds it, the slice as the form sends it: a
    // Timestamp's distance back from the record's time in the short and medium forms, and anything else as stored.
    // One expression, so that a record's size costs no function per piece.
    template <Width kWidth, size_t W, typename Record, size_t... J>
    static FORCE_INLINE uint32_t built_word(const Record& record, std::index_sequence<J...>) {
        constexpr const WordPieces& kWord = kWordPieces<Payload, kWidth>[W];
        return (
            (uint32_t(
                 uint64_t(
                     kIsDelta<kWord.pieces[J].slice> && kWidth != Width::Bits64
                         ? uint64_t(record.time_lo - uint32_t(record.stamps[kWord.pieces[J].slice]))
                         : uint64_t(slice_bits<Payload, kWord.pieces[J].slice>(record.source))) >>
                 (kWord.pieces[J].high_half ? 32 : 0))
             << kWord.pieces[J].shift) |
            ... | 0u);
    }
};

// The medium form's payload word A, from its slices as their arguments store them.
template <typename Payload, size_t A, typename Source, size_t... J>
FORCE_INLINE uint32_t stored_word(const Source& source, std::index_sequence<J...>) {
    constexpr const WordPieces& kWord = kWordPieces<Payload, streaming_profiler::detail::TimestampWidth::Bits32>[A];
    return (
        (uint32_t(uint64_t(slice_bits<Payload, kWord.pieces[J].slice>(source)) >> (kWord.pieces[J].high_half ? 32 : 0))
         << kWord.pieces[J].shift) |
        ... | 0u);
}
// Register word A of a record that crosses an out-of-line writer's call in registers (RegisterWords).
template <typename Payload, size_t A, typename Source>
FORCE_INLINE uint32_t register_word(const Source& source) {
    if constexpr (!kSendsDeltas<Payload>) {
        return stored_word<Payload, A>(
            source, WordPieceIndices<Payload, streaming_profiler::detail::TimestampWidth::Bits32, A>{});
    } else {
        constexpr size_t slice = kRegisterWordSlice<Payload>[A];
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
    if constexpr (kReadsArgumentsInPlace<Payload>) {
        publish_with<RecordWriter<Payload, Source>, kImmediately, kNoInline>(header, &source);
    } else {
        publish_with<RecordWriter<Payload, RegisterWords<Payload>>, kImmediately, kNoInline>(
            header, register_word<Payload, A>(source)...);
    }
}

}  // namespace kernel_profiler::records

namespace kernel_profiler {

static_assert(
    records::kLayout<records::PublishTimePayload<&streaming_profiler::Interval::end>, streaming_profiler::detail::TimestampWidth::Bits64>.packet_words == MIN_RESERVE_WORDS,
    "every room check reserves a zone record's longest packet");

// Whether ppfmt's stall record, which write_pending_stall writes and the host describes stalls from, is the medium
// record DevicePublish lays out for a zone, whose Interval's Timestamps lie back to back.
constexpr bool stall_record_is_zone_medium_record() {
    using Payload = records::PublishTimePayload<&streaming_profiler::Interval::end>;
    constexpr auto layout = records::kLayout<Payload, streaming_profiler::detail::TimestampWidth::Bits32>;
    return layout.packet_words == ppfmt::STALL_PACKET_WORDS && layout.slices.size() == 1 && layout.slices[0].to == 0 &&
           8u * layout.slices[0].bytes == ppfmt::STALL_START_DELTA_BITS &&
           records::kUnsentLeaf<Payload> == ppfmt::STALL_PUBLISH_TIME_FIELD &&
           offsetof(streaming_profiler::Interval, end) ==
               streaming_profiler::detail::code_bytes(streaming_profiler::detail::kTimestampCode);
}
static_assert(stall_record_is_zone_medium_record(), "the host describes stalls by ppfmt's stall record");

}  // namespace kernel_profiler

#endif
