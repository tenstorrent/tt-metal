// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side (zone id -> source location) table for the streaming device profiler, harvested from the
// .tt_zone_meta / .tt_zone_str sections of each kernel/firmware ELF.
//
// Filled from llrt::get_risc_binary(), the one point every device-executed binary passes through, so an id
// is always registered before the kernel that emits it can run. Consumers read it by delta rather than by
// snapshot, because a model streams zones from running kernels while later kernels are still JIT-compiling.
// Never persisted: structural ids change between builds, so a cached table would mix generations.
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

    // Read .tt_zone_meta out of `elf_path` and register its records. Idempotent per path. Never throws: a
    // missing file, missing section or malformed record is a warning at most, since failing to name a zone
    // must not fail a run.
    void ingest_elf(const std::string& elf_path);

    // Copy the entries at or after `from`, an index into an append-only log, into `out`; returns the new
    // cursor for the caller to keep.
    uint32_t additions_since(uint32_t from, std::vector<ZoneMetaEntry>& out) const;

    // Ids registered twice with different source locations. Expected zero: structural ids only collide if
    // two translation units were handed the same tu_id (get_or_assign_profiler_tu_id in jit_build/build.cpp).
    uint64_t collisions() const;

    // Ingested ELF count, total record count, and ELFs whose .tt_zone_meta failed a format guard (a stale
    // record layout left behind in the JIT cache).
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
