// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's call sites, read from each device image as it loads. Each image that loads gets the next
// block of the process's call-site ids, so an id names one call site in the process, and that site is known before the
// image can run. Ids are assigned in load order and never persisted.
#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "hostdev/record_site.h"

namespace ll_api {
class ElfFile;
}

namespace tt::llrt {

struct FieldBits {
    uint32_t bit = 0, width = 0;
    bool operator==(const FieldBits&) const = default;
};

// Where a field lies, from a sample that sets only its bits: the run from the lowest set bit to the highest.
FieldBits find_set_bits(std::span<const std::byte> sample);

// A call site as its image describes it, every string in it lasting for the process.
struct CallSite {
    tt::tt_metal::experimental::streaming_profiler::SourceLocation location;
    std::string_view metadata_type;
    std::vector<uint16_t> payload_codes, metadata_codes;
    std::vector<FieldBits> metadata_bits;  // each metadata field's bits in the metadata struct
    // An argument the call passes, a payload struct being one.
    struct Argument {
        std::string_view type;  // empty for a plain argument
        uint32_t first_field = 0;
        bool is_array = false;
        std::vector<FieldBits> field_bits;  // where its fields lie in it, in one element for an array
    };
    std::vector<Argument> arguments;
    std::vector<uint32_t> argument_field_counts;  // each argument's payload field count, every element's for an array
    // The packet layout of a form its records take, which their body length tells apart.
    struct PacketLayout {
        tt::tt_metal::experimental::streaming_profiler::detail::TimeForm time_form{};
        uint32_t body_words = 0;
        std::vector<FieldBits> payload_bits;  // each payload field's bits in the packet's payload, width 0 if not sent
    };
    std::vector<PacketLayout> packet_layouts;
    bool timestamps_as_distances = false;  // whether its Timestamp fields travel back from the record's time
    // The payload field that is the record's own time, which no form sends.
    uint32_t payload_publish_time_field =
        tt::tt_metal::experimental::streaming_profiler::detail::kNoPayloadPublishTimeField;
    // The metadata's fields as the call site sets them: each one's bits, sign-extended for a signed field, or a
    // String's text.
    std::vector<std::variant<uint64_t, std::string_view>> metadata;
};

// Gives the image at `elf_path` its block of call-site ids, rebases `elf` there and reads its call sites. Does nothing
// for an ELF without call sites. Throws when the id space runs out, and when a call site sends a type whose fields
// contradict another showing of its name.
void load_call_sites(const std::string& elf_path, ll_api::ElfFile& elf);

// The call site with id `id`, or null when no loaded image holds it. A call site lives for the process.
const CallSite* find_call_site(uint32_t id);

// Notes the field codes of a type a host callback reads. Throws when kernels show the same name with other fields.
void add_host_type(std::string_view name, std::span<const uint16_t> codes);

}  // namespace tt::llrt
