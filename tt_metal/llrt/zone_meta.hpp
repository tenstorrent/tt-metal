// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side zone id -> source location table for the streaming profiler, built as each ELF loads
// (ll_api::memory, llrt/tt_memory.cpp). An image links its zone ids densely from one fixed VMA
// (hostdev/profiler_zone_id.h); here it is given the next block of the process-wide 16-bit id space, its ids
// are rebased there inside the loaded image (ElfFile::RebaseZoneIds), and its .tt_zone_meta / .tt_zone_str
// records are harvested with the rebased ids. An id is therefore unique across every image of this process
// by construction, and named strictly before the kernel that emits it can run. Handed to its listener as it
// grows, since kernels stream zones while later ones are still compiling. Never persisted: ids are assigned
// in load order.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace ll_api {
class ElfFile;
}

namespace tt::llrt {

struct ZoneMetaEntry {
    uint32_t zone_id = 0;
    std::string name;
    std::string file;
    uint32_t line = 0;
};

class ZoneMetaRegistry {
public:
    static ZoneMetaRegistry& instance();

    // Gives the image at PATH its id block and rebases ELF there; harvests its names on the first call for a
    // path (a later call for the same path rebases to the same block and registers nothing). A malformed
    // section is reported, not thrown: failing to name a zone must not fail a run. Throws only when the id
    // space is exhausted.
    void ingest_elf(const std::string& elf_path, ll_api::ElfFile& elf);

    // One listener, called under the registry's lock with each ELF's new entries as they register, after a replay of
    // every entry already registered. Entries never move or die.
    using Listener = std::function<void(std::span<const ZoneMetaEntry* const>)>;
    void set_listener(Listener listener);

    // Ids handed out so far; the next image's block starts here and the space ends at TT_ZONE_STALL_ID.
    uint32_t ids_assigned() const;
    // Records whose id fell outside their image's block or repeated one; their zones render as Zone_<id>.
    uint64_t malformed_records() const;
    // ELFs whose .tt_zone_meta failed a format guard (a stale layout in the JIT cache).
    uint64_t foreign_sections() const;

private:
    ZoneMetaRegistry() = default;
};

}  // namespace tt::llrt
