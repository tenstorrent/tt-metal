// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A DevicePublish call site's entry in a kernel ELF, written by internal/debug/device_publish.h and read when
// the ELF loads (llrt/call_sites.cpp). Each struct comes with one sample per field, a copy of the struct with that
// field's bits set, so the host finds every field where the device compiler put it, and each of the call site's forms
// lists what its payload holds of each value.
#pragma once

#include <cstddef>
#include <cstdint>

#include "hostdevcommon/streaming_profiler_reflection.hpp"

namespace tt::tt_metal::experimental::streaming_profiler::detail {

// An address in a section the device never loads: in kernels a pointer the linker fills in, on the host the 32-bit
// link address the loader resolves by the section's own address.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
template <typename T>
using SectionAddress = const T*;
#else
template <typename T>
using SectionAddress = uint32_t;
#endif

// `bytes` bytes of value `value` from byte `from` on, at byte `to` of a record's payload.
struct RecordCopy {
    uint16_t value, from, to, bytes;
};

// A value a call passes, the data struct being one: its type's name for a struct or an enum, its field count, how many
// elements it has as an array (1 otherwise), its struct's samples and its size.
struct RecordValue {
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

// One form: its packet's body words, and what its payload holds of the values' fields, each Timestamp that travels as
// a distance back from the record's time in the form's bytes for it.
struct RecordForm {
    TimeForm time_form;
    uint32_t body_words;
    SectionAddress<RecordCopy> copies;
    uint32_t copy_count;
};

struct RecordSiteEntry {
    SectionAddress<char> id;  // the call site's handle in .tt_record_ids, which the loader rebases into an id
    SectionAddress<char> metadata_type;
    SectionAddress<char> file;
    uint32_t line;
    SectionAddress<char> function;
    SectionAddress<void> metadata;
    uint32_t metadata_size;
    SectionAddress<void> metadata_samples;
    SectionAddress<uint16_t> metadata_codes;
    uint32_t metadata_field_count;
    SectionAddress<uint16_t> data_codes;
    uint32_t data_field_count;
    SectionAddress<RecordValue> data_values;
    uint32_t data_value_count;
    SectionAddress<RecordForm> data_forms;  // one per form its records take, shortest first
    uint32_t data_form_count;
    uint32_t data_published_field;  // the data field that is the record's publish time and isn't sent
    SectionAddress<void> padding_canary;
};

// A RecordSiteEntry's data_published_field when no field is the record's publish time.
inline constexpr uint32_t kNoPublishedField = ~0u;

// Both compilers lay these tables out alike.
static_assert(sizeof(RecordCopy) == 4 * sizeof(uint16_t));
static_assert(sizeof(RecordValue) == 6 * sizeof(uint32_t));
static_assert(sizeof(RecordForm) == 4 * sizeof(uint32_t));
static_assert(sizeof(RecordSiteEntry) == 18 * sizeof(uint32_t));

// A value's size on the device, which is also its alignment.
constexpr uint32_t code_bytes(uint16_t code) { return (code & 0xff) / 8; }

inline constexpr uint16_t kStringCode = type_code<String>();
inline constexpr uint16_t kTimestampCode = type_code<Timestamp>();

// The text a String field's sample holds, which the host finds the field by.
inline constexpr char kStringSample[] = "tt-record-string-sample";

// The device's padding canary (internal/debug/device_publish.h) as RV32 lays it out: a uint8_t, its padding, a String's
// four bytes, 3 and 10 bits of bit-fields that leave three bits of their second byte as padding, a uint8_t and the tail
// padding. With its fields all ones, its bytes are these value bits, the String's bytes holding an address.
inline constexpr std::array<uint8_t, 12> kPaddingCanaryValueBits = {
    0xFF, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x1F, 0xFF, 0};
inline constexpr uint32_t kPaddingCanaryStringAt = 4;

}  // namespace tt::tt_metal::experimental::streaming_profiler::detail
