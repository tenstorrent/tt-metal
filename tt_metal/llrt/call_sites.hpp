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
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "hostdev/record_site.h"

namespace ll_api {
class ElfFile;
}

namespace tt::llrt {

// An image's .tt_record_data section, which holds every string its call sites name, and the section's link address.
struct CallSiteStrings {
    std::vector<char> text;
    uint64_t vma = 0;

    std::string_view string_at(uint32_t address) const {
        const uint64_t offset = uint64_t{address} - vma;
        if (offset >= text.size()) {
            return {};
        }
        const std::string_view rest(text.data() + offset, text.size() - offset);
        return rest.substr(0, rest.find('\0'));
    }
};

struct FieldBits {
    uint32_t bit = 0, width = 0;
    bool operator==(const FieldBits&) const = default;
};

// Where a field lies, from a sample that sets only its bits: the run from the lowest set bit to the highest.
FieldBits find_set_bits(std::span<const std::byte> sample);

struct CallSite {
    tt::tt_metal::experimental::streaming_profiler::SourceLocation location;
    std::string_view metadata_type;
    std::vector<uint16_t> data_codes, metadata_codes;
    std::vector<FieldBits> metadata_bits;  // each metadata field's bits in the metadata struct
    // A value the call passes, struct data being one.
    struct Value {
        std::string_view type;  // empty for a plain value
        uint32_t first_field = 0;
        bool is_array = false;
        std::vector<FieldBits> field_bits;  // where its fields lie in it, in one element for an array
    };
    std::vector<Value> values;
    std::vector<uint32_t> value_fields;
    // A form its records take, which their body length tells apart.
    struct Form {
        tt::tt_metal::experimental::streaming_profiler::detail::TimeForm time_form{};
        uint32_t body_words = 0;
        std::vector<FieldBits> data_bits;  // each data field's bits in the payload, width 0 if it holds none
    };
    std::vector<Form> forms;
    bool timestamps_as_distances = false;  // whether its Timestamp fields travel back from the record's time
    // The data field that is the record's own time, which no form sends.
    uint32_t published_field = tt::tt_metal::experimental::streaming_profiler::detail::kNoPublishedField;
    std::vector<std::byte> metadata;  // the metadata struct, as the device compiler laid it out
    std::shared_ptr<const CallSiteStrings> strings;
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
