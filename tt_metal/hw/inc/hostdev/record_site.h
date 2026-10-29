// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The descriptor of each DevicePublish call site in a kernel's ELF, which internal/debug/device_publish_impl.h writes
// and llrt/call_sites.cpp reads when the ELF loads. Each struct comes with its fields' bits, which the device compiler
// reads from samples of the struct, so the host learns where that compiler put each field.
#pragma once

#include <cstddef>
#include <cstdint>

#include "hostdevcommon/streaming_profiler_reflection.hpp"

#ifndef TT_DEVICE_COMPILE
#include <array>
#include <cstring>
#endif

namespace tt::tt_metal::experimental::streaming_profiler::detail {

// An address in a section the device never loads: in kernels, a pointer the linker fills in; on the host, the 32-bit
// link address, which the loader resolves against the section's own address.
#ifdef TT_DEVICE_COMPILE
template <typename T>
using SectionAddress = const T*;
#else
template <typename T>
using SectionAddress = uint32_t;
#endif

// A piece of a call's argument in a record's payload: `bytes` bytes from byte `from` of argument `argument`, at byte
// `to` of the payload.
struct ArgumentSlice {
    uint16_t argument, from, to, bytes;
};

// Where a field lies in its struct: the bits from the lowest its sample sets to the highest.
struct FieldBits {
    uint32_t bit = 0, width = 0;
#ifndef TT_DEVICE_COMPILE
    bool operator==(const FieldBits&) const = default;
#endif
};

// A struct's samples number its fields in binary. Sample k sets each field whose number plus one has bit k set, to a
// value that sets at least the field's lowest and highest bits, and leaves every other bit zero, padding included. Read
// down one bit, the samples spell the number plus one of the field that owns it, or 0 for a bit no field owns. A struct
// of `field_count` fields takes this many samples.
constexpr uint32_t sample_count(uint32_t field_count) {
    uint32_t count = 0;
    while ((field_count >> count) != 0) {
        ++count;
    }
    return count;
}
// Each field's bits, from `count` samples of a `size`-byte struct, back to back.
template <typename Byte>
constexpr void field_bits_from_samples(const Byte* samples, uint32_t count, uint32_t size, FieldBits* fields) {
    for (uint32_t bit = 0; bit < 8 * size; ++bit) {
        uint32_t number = 0;
        for (uint32_t k = 0; k < count; ++k) {
            number |= ((uint32_t(samples[k * size + bit / 8]) >> (bit % 8)) & 1u) << k;
        }
        if (number != 0) {
            FieldBits& field = fields[number - 1];
            if (field.width == 0) {
                field.bit = bit;
            }
            field.width = bit + 1 - field.bit;
        }
    }
}

// One argument of a DevicePublish call, a payload struct counting as one: its type's name for a struct or an enum, its
// field count, its element count as an array (1 otherwise), its struct's field bits, and its size.
struct CallArgument {
    SectionAddress<char> type;
    uint32_t field_count;
    uint32_t count;
    uint32_t is_array;
    SectionAddress<FieldBits> field_bits;
    uint32_t size;
};

// The width of every Timestamp in a record's packet: its publish Timestamp, taken when DevicePublish is called, and any
// Timestamp fields of its payload. Payload Timestamps travel as offsets back from the publish Timestamp, except 64-bit
// ones, which travel whole. A 16-bit publish Timestamp is an offset from the cursor, the publish Timestamp of the
// RISC's last record from a call site with a short form, which the decoder tracks too; a 32-bit one is the low half,
// whose high half comes from the RISC's last timer packet. The packet's length tells the forms apart, and a form's
// value is how many time words its packets carry. A record takes the shortest form that fits. Times assume a 1.35 GHz
// device clock.
enum class TimestampWidth : uint32_t {
    // 16-bit Timestamps: used for a record with payload Timestamps whose publish Timestamp is less than ~49 µs after
    // the RISC's previous record and whose payload Timestamps are less than ~49 µs before its publish Timestamp.
    Bits16,
    // 32-bit Timestamps: used for a record whose publish Timestamp is any time after the RISC's previous record and
    // whose payload Timestamps are less than ~3 s before its publish Timestamp.
    Bits32,
    // 64-bit Timestamps: used for a record whose publish Timestamp is any time after the RISC's previous record and
    // whose payload Timestamps are any time before its publish Timestamp, and for every record sent to the external
    // process.
    Bits64,
};

// One form's packet: its length in words after word 0, and its slices, which say where each piece of each argument sits
// in the payload. A Timestamp's slice is 2, 4 or 8 bytes, by form.
struct PacketLayout {
    TimestampWidth timestamp_width;
    uint32_t packet_words;  // word 0 included
    SectionAddress<ArgumentSlice> slices;
    uint32_t slice_count;
};

// What the host knows of one DevicePublish call site, one per call site in .device_publish_descriptors. `file` comes
// first, so every 4-byte field after it packs without padding.
struct CallSiteDescriptor {
    uint64_t file;            // the String of the call site's file name
    SectionAddress<char> id;  // the call site's handle in .device_publish_ids, which the loader rebases into an id
    uint32_t line;
    SectionAddress<char> function;

    SectionAddress<char> metadata_type;
    SectionAddress<void> metadata;
    uint32_t metadata_size;
    SectionAddress<FieldBits> metadata_field_bits;
    SectionAddress<uint16_t> metadata_codes;
    uint32_t metadata_field_count;

    SectionAddress<uint16_t> payload_codes;
    uint32_t payload_field_count;
    SectionAddress<CallArgument> payload_arguments;
    uint32_t payload_argument_count;
    uint32_t payload_publish_time_field;  // the payload field that is the record's publish time and isn't sent

    SectionAddress<PacketLayout> packet_layouts;  // one per form its records take, shortest first
    uint32_t packet_layout_count;
};

// A CallSiteDescriptor's payload_publish_time_field when no field is the record's publish time.
inline constexpr uint32_t kNoPayloadPublishTimeField = ~0u;

// An entry of an image's .device_publish_strings: a String's hash, and its text of `size` bytes followed by a NUL. Each
// translation unit that uses a text adds its own entry for it.
struct StringEntry {
    uint64_t hash;
    SectionAddress<char> text;
    uint32_t size;
};

// Both compilers lay these tables out alike.
static_assert(sizeof(ArgumentSlice) == 4 * sizeof(uint16_t));
static_assert(sizeof(FieldBits) == 2 * sizeof(uint32_t));
static_assert(sizeof(CallArgument) == 6 * sizeof(uint32_t));
static_assert(sizeof(PacketLayout) == 4 * sizeof(uint32_t));
static_assert(sizeof(CallSiteDescriptor) == 18 * sizeof(uint32_t));
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

}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
