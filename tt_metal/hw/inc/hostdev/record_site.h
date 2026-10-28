// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A DevicePublish call site's descriptor in a kernel ELF, written by internal/debug/device_publish_impl.h and read when
// the ELF loads (llrt/call_sites.cpp). Each struct comes with one sample per field, a copy of the struct with that
// field's bits set, so the host finds every field where the device compiler put it, and each of the call site's packet
// layouts lists where its payload holds each argument's slices.
#pragma once

#include <cstddef>
#include <cstdint>

#include "hostdevcommon/device_observability_reflection.hpp"

#ifndef TT_DEVICE_COMPILE
#include <array>
#include <cstring>
#endif

namespace tt::tt_metal::experimental::device_observability::detail {

// An address in a section the device never loads: in kernels a pointer the linker fills in, on the host the 32-bit
// link address the loader resolves by the section's own address.
#ifdef TT_DEVICE_COMPILE
template <typename T>
using SectionAddress = const T*;
#else
template <typename T>
using SectionAddress = uint32_t;
#endif

// `bytes` bytes of argument `argument` from byte `from` on, at byte `to` of a record's payload.
struct ArgumentSlice {
    uint16_t argument, from, to, bytes;
};

// An argument a call passes, the payload struct being one: its type's name for a struct or an enum, its field count,
// how many elements it has as an array (1 otherwise), its struct's samples and its size.
struct CallArgument {
    SectionAddress<char> type;
    uint32_t field_count;
    uint32_t count;
    uint32_t is_array;
    SectionAddress<void> samples;
    uint32_t size;
};

// Every record carries its own time, in one of the forms its packet's length tells apart. Only the short and medium
// records of a call site with a short form move the lane's cursor, so the decoder moves its own by the call site's
// forms. A form's value is its packets' count of time words, so their order is fixed.
enum class TimeForm : uint32_t {
    Short,   // no time word: the time is a 16-bit distance from the lane's cursor, and each Timestamp a 16-bit distance
    Medium,  // the time's low word, and each Timestamp a 32-bit distance
    Full,    // the time whole, and each Timestamp a 64-bit distance
};

// The packet of one form: its body words, and where its payload holds each argument's slices, each Timestamp that
// travels as a distance back from the record's time in the form's bytes for it.
struct PacketLayout {
    TimeForm time_form;
    uint32_t body_words;
    SectionAddress<ArgumentSlice> slices;
    uint32_t slice_count;
};

struct CallSiteDescriptor {
    SectionAddress<char> id;  // the call site's handle in .device_publish_ids, which the loader rebases into an id
    SectionAddress<char> metadata_type;
    uint64_t file;  // the String of the call site's file name
    uint32_t line;
    SectionAddress<char> function;
    SectionAddress<void> metadata;
    uint32_t metadata_size;
    SectionAddress<void> metadata_samples;
    SectionAddress<uint16_t> metadata_codes;
    uint32_t metadata_field_count;
    SectionAddress<uint16_t> payload_codes;
    uint32_t payload_field_count;
    SectionAddress<CallArgument> payload_arguments;
    uint32_t payload_argument_count;
    SectionAddress<PacketLayout> packet_layouts;  // one per form its records take, shortest first
    uint32_t packet_layout_count;
    uint32_t payload_publish_time_field;  // the payload field that is the record's publish time and isn't sent
    // 1 when the call site's records go to the external process too: they are always full, and their Timestamps whole.
    uint32_t sends_to_external_process;
};

// A CallSiteDescriptor's payload_publish_time_field when no field is the record's publish time.
inline constexpr uint32_t kNoPayloadPublishTimeField = ~0u;

// An entry of an image's .device_publish_strings: a String's hash and its text, `size` bytes before the NUL that ends
// it. Every translation unit that uses a text has an entry of its own for it.
struct StringEntry {
    uint64_t hash;
    SectionAddress<char> text;
    uint32_t size;
};

// Both compilers lay these tables out alike.
static_assert(sizeof(ArgumentSlice) == 4 * sizeof(uint16_t));
static_assert(sizeof(CallArgument) == 6 * sizeof(uint32_t));
static_assert(sizeof(PacketLayout) == 4 * sizeof(uint32_t));
static_assert(sizeof(CallSiteDescriptor) == 20 * sizeof(uint32_t));
static_assert(sizeof(StringEntry) == 4 * sizeof(uint32_t));

// A value's size on the device, which is also its alignment.
constexpr uint32_t code_bytes(uint16_t code) { return (code & 0xff) / 8; }

inline constexpr uint16_t kStringCode = type_code<String>();
inline constexpr uint16_t kTimestampCode = type_code<Timestamp>();

#ifndef TT_DEVICE_COMPILE
template <typename T>
T load_unaligned(const std::byte* at) {
    T value;
    std::memcpy(&value, at, sizeof(value));
    return value;
}

// The `width` bits from bit `bit` of `bytes`, sign-extended for a signed field.
inline uint64_t field_value(const std::byte* bytes, uint32_t bit, uint32_t width, uint16_t code) {
    const std::byte* const at = bytes + bit / 8;
    const uint32_t shift = bit % 8;
    uint64_t value = 0;
    if (shift == 0 && width == 8 * code_bytes(code)) {
        // A field in whole bytes of its own size, as most are, loads just those bytes.
        switch (width) {
            case 8: value = std::to_integer<uint8_t>(*at); break;
            case 16: value = load_unaligned<uint16_t>(at); break;
            case 32: value = load_unaligned<uint32_t>(at); break;
            default: value = load_unaligned<uint64_t>(at); break;
        }
    } else {
        std::array<std::byte, 16> window{};
        std::memcpy(window.data(), at, (shift + width + 7) / 8);
        uint64_t low = 0, high = 0;
        std::memcpy(&low, window.data(), sizeof(low));
        std::memcpy(&high, window.data() + sizeof(low), sizeof(high));
        value = shift == 0 ? low : low >> shift | high << (64 - shift);
    }
    if (width < 64) {
        const uint64_t mask = (uint64_t{1} << width) - 1;
        const bool negative = (code >> 8) == 'i' && (value >> (width - 1) & 1) != 0;
        value = negative ? value | ~mask : value & mask;
    }
    return value;
}
#endif

}  // namespace tt::tt_metal::experimental::device_observability::detail
