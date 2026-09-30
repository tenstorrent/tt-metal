// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side zone id -> source location table, harvested from each ELF's .tt_zone_meta / .tt_zone_str. Filled
// from llrt::get_risc_binary(), which every device-executed binary passes through, so an id is registered
// before the kernel that emits it can run. Handed to its listener as it grows, since kernels stream zones while
// later ones are still compiling. Never persisted: structural ids change between builds.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>

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

    // One listener, called under the registry's lock with each ELF's new entries as they register, after a replay of
    // every entry already registered. Entries never move or die.
    using Listener = std::function<void(std::span<const ZoneMetaEntry* const>)>;
    void set_listener(Listener listener);

    // Ids registered twice with different locations; nonzero means two TUs got the same tu_id.
    uint64_t collisions() const;
    // ELFs whose .tt_zone_meta failed a format guard (a stale layout in the JIT cache).
    uint64_t foreign_sections() const;

private:
    ZoneMetaRegistry() = default;
};

}  // namespace tt::llrt
