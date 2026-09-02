// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side (zone id -> source location) table for the STREAMING device profiler, harvested from the
// .tt_zone_meta / .tt_zone_str sections of each kernel/firmware ELF as it is loaded.
//
// WHY THIS LIVES HERE, AND WHY IT IS PER-ELF. The DRAM profiler learns zone names by grepping
// `#pragma message` lines out of the JIT build log, and the streaming consumer used to borrow that: one
// std::call_once snapshot on first drain. That is wrong for a model, not just imprecise -- a workload
// streams zones from kernels that are already running while LATER kernels are still JIT-compiling, so a
// single snapshot is taken when the name table is a fraction of its final size.
//
// So the table is fed incrementally, from the one place every device-executed binary funnels through:
// llrt::get_risc_binary(), which opens each ELF exactly once per path per process. A kernel cannot emit
// a marker before its binary has been loaded, so by construction every id that reaches the consumer was
// registered first. No snapshot, no ordering assumption, nothing to get stale.
//
// The consumer reads the table by DELTA (see additions_since) rather than by copying it, because a
// version-bump-and-recopy would be O(kernels * zones) over a model run.
//
// NOT PERSISTED. Structural ids legitimately change between builds -- a source line shifts, or the tu-id
// registry is compacted -- so a cached id->name file would mix generations and report stale entries as
// collisions. The table is rebuilt from the current run's ELFs every time, and dies with the process.
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

    // Read .tt_zone_meta out of `elf_path` and register everything in it. Idempotent per path: a path
    // already ingested returns immediately. Never throws -- a missing file, a missing section or a
    // malformed record is a warning at most, because failing to NAME a zone must never fail a run.
    void ingest_elf(const std::string& elf_path);

    // Entries registered at or after `from`. `from` is an index into a monotonically growing append-only
    // log, so a consumer keeps its own cursor and copies only what is new. Returns the new cursor.
    uint32_t additions_since(uint32_t from, std::vector<ZoneMetaEntry>& out) const;

    // Distinct ids that were registered twice with DIFFERENT source locations. Zero is the invariant:
    // structural ids are collision-free by construction, so any count here means two translation units
    // were handed the same tu_id (see get_or_assign_profiler_tu_id in jit_build/build.cpp) -- reported
    // rather than silently letting one name win.
    uint64_t collisions() const;

    // Ingested ELF count, total record count, and the number of ELFs whose .tt_zone_meta failed a
    // format guard (a foreign/stale record layout left behind in the JIT cache), for the teardown summary.
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
