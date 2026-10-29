// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's call sites, read from each device image as it loads. Each image that loads gets the next
// block of the process's call-site ids, so an id names one call site in the process, and that site is known before the
// image can run. Ids are assigned in load order and never persisted. The external process holds each Metal process's
// call sites apart, under that process's ids (CallSites).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
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

using FieldBits = tt::tt_metal::experimental::streaming_profiler::detail::FieldBits;

// A call site as its image describes it, every string in it lasting for the process.
struct CallSite {
    tt::tt_metal::experimental::streaming_profiler::SourceLocation location;
    std::string_view metadata_type;
    std::vector<uint16_t> metadata_codes;
    std::vector<FieldBits> metadata_bits;  // each metadata field's bits in the metadata struct
    // The metadata's fields as the call site sets them: each one's bits, sign-extended for a signed field, or a
    // String's text.
    std::vector<std::variant<uint64_t, std::string_view>> metadata;

    std::vector<uint16_t> payload_codes;
    // An argument the call passes, a payload struct being one.
    struct Argument {
        std::string_view type;  // empty for a plain argument
        uint32_t first_field = 0;
        bool is_array = false;
        std::vector<FieldBits> field_bits;  // where its fields lie in it, in one element for an array
    };
    std::vector<Argument> arguments;
    std::vector<uint32_t> argument_field_counts;  // each argument's payload field count, every element's for an array
    // The payload field that is the record's own time, which no form sends.
    uint32_t payload_publish_time_field =
        tt::tt_metal::experimental::streaming_profiler::detail::kNoPayloadPublishTimeField;

    // The packet layout of a form its records take, which their body length tells apart.
    struct PacketLayout {
        tt::tt_metal::experimental::streaming_profiler::detail::TimestampWidth timestamp_width{};
        uint32_t packet_words = 0;            // word 0 included
        std::vector<FieldBits> payload_bits;  // each payload field's bits in the packet's payload, width 0 if not sent
    };
    std::vector<PacketLayout> packet_layouts;
    bool timestamps_as_deltas = false;  // whether its short and medium forms send Timestamps as distances
};

}  // namespace tt::llrt

namespace tt::tt_metal::experimental::streaming_profiler::detail {

// A Metal process's call sites, by the ids its images gave them: this process's own (tt::llrt::process_call_sites()),
// which its images add as they load, or, in the external process, a Metal process's that it receives from, which that
// process sends. Each holds the stall site, which every image shares and none describes, at TT_ZONE_STALL_ID. A
// site's strings are stored for the life of the process; the rest of it goes with its call sites.
class CallSites {
public:
    CallSites();
    ~CallSites();
    CallSites(const CallSites&) = delete;
    CallSites& operator=(const CallSites&) = delete;

    // Makes `site` the one with id `id`, its strings stored. Throws when it sends a type whose fields differ from
    // another showing of its name, here or in a host callback (tt::llrt::add_host_type).
    void add(uint32_t id, tt::llrt::CallSite site);
    // The site with id `id`, or null when none has it.
    const tt::llrt::CallSite* find(uint32_t id) const { return by_id_[id].load(std::memory_order_acquire); }
    // The sites' metadata, by id, each built the first time a record asks for it.
    std::atomic<const void*>* metadata() const { return metadata_.get(); }
    // Site `id`'s metadata, which `build` makes the first time anyone asks, and `destroy` deletes with these call
    // sites.
    const void* build_metadata(
        uint32_t id,
        const std::function<const void*(const tt::llrt::CallSite& site)>& build,
        void (*destroy)(const void* metadata)) const;
    // Throws when a site here sends a type named `name` with fields other than `codes`.
    void check_host_type(std::string_view name, std::span<const uint16_t> codes) const;

private:
    struct State;
    std::unique_ptr<State> state_;
    std::unique_ptr<std::atomic<const tt::llrt::CallSite*>[]> by_id_;
    std::unique_ptr<std::atomic<const void*>[]> metadata_;
};

}  // namespace tt::tt_metal::experimental::streaming_profiler::detail

namespace tt::llrt {

using CallSites = tt::tt_metal::experimental::streaming_profiler::detail::CallSites;

// This process's call sites: those of the images it loads.
CallSites& process_call_sites();

// Gives the image at `elf_path` its block of this process's call-site ids, rebases `elf` there and adds its call sites
// to process_call_sites(). Does nothing for an ELF without call sites. Throws when the id space runs out, and when a
// call site sends a type whose fields contradict another showing of its name.
void load_call_sites(const std::string& elf_path, ll_api::ElfFile& elf);

// Notes the field codes of a type a host callback reads, which every call site's showing of the name must match.
// Throws when this process's kernels show it with other fields.
void add_host_type(std::string_view name, std::span<const uint16_t> codes);

}  // namespace tt::llrt
