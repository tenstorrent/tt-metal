// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side zone id -> source location table, harvested from each ELF's .tt_zone_meta / .tt_zone_str. Filled
// from llrt::get_risc_binary(), which every device-executed binary passes through, so an id is registered
// before the kernel that emits it can run. Read by delta, since kernels stream zones while later ones are
// still compiling. Never persisted: structural ids change between builds.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

    // Idempotent per path. Never throws: failing to name a zone must not fail a run.
    void ingest_elf(const std::string& elf_path);

    // Copy the entries at or after `from` into `out`; returns the new cursor.
    uint32_t additions_since(uint32_t from, std::vector<ZoneMetaEntry>& out) const;

    // Ids registered twice with different locations; nonzero means two TUs got the same tu_id.
    uint64_t collisions() const;

    // Ingested ELFs, records, and ELFs whose .tt_zone_meta failed a format guard (a stale layout in the JIT
    // cache).
    struct Stats {
        uint64_t elfs = 0;
        uint64_t records = 0;
        uint64_t foreign_sections = 0;
    };
    Stats stats() const;

private:
    ZoneMetaRegistry() = default;
};

}  // namespace tt::llrt
