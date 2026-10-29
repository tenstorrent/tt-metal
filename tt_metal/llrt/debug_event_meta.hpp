// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-side debug event id -> source location table, harvested from each ELF's .tt_debug_event_meta /
// .tt_debug_event_str. Filled from llrt::get_risc_binary(), which every device-executed binary passes through, so an id
// is registered before the kernel that emits it can run. Handed to its listener as it grows, since kernels stream
// events while later ones are still compiling. Never persisted: structural ids change between builds.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace ll_api {
class ElfFile;
}

namespace tt::llrt {

struct DebugEventMetaEntry {
    uint32_t event_id = 0;
    std::string name;
    std::string file;
    uint32_t line = 0;
};

class DebugEventMetaRegistry {
public:
    static DebugEventMetaRegistry& instance();

    // Harvest event names from an already-open ELF. `elf_path` is the idempotency key and diagnostic label only.
    // A no-op for an ELF without .tt_debug_event_meta, so it can be called for every binary. Idempotent per path.
    // Never throws: failing to name an event must not fail a run.
    void ingest_elf(const ll_api::ElfFile& elf, const std::string& elf_path);

    // One listener, called under the registry's lock with each ELF's new entries as they register, after a replay of
    // every entry already registered. Entries never move or die.
    using Listener = std::function<void(std::span<const DebugEventMetaEntry* const>)>;
    void set_listener(Listener listener);

    // Ids registered twice with different locations; nonzero means two TUs got the same tu_id.
    uint64_t collisions() const;
    // ELFs whose .tt_debug_event_meta failed a format guard (a stale layout in the JIT cache).
    uint64_t foreign_sections() const;

private:
    DebugEventMetaRegistry() = default;
};

}  // namespace tt::llrt
