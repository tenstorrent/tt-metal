// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "observability_event_meta.hpp"

#include <cstring>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/observability_event_id.h"
#include "tt_elffile.hpp"

namespace tt::llrt {

namespace {

// Mirrors TT_OBSERVABILITY_EVENT_DEFINE_ID; the host walks the section at a fixed 16-byte stride.
struct ObservabilityEventMetaRecord {
    uint32_t event_id;
    uint32_t name_ptr;  // VMA into .tt_observability_event_str
    uint32_t file_ptr;  // VMA into .tt_observability_event_str
    uint32_t line;
};
static_assert(
    sizeof(ObservabilityEventMetaRecord) == TT_OBSERVABILITY_EVENT_META_RECORD_BYTES,
    "the device emitter writes exactly four .long fields");

struct State {
    mutable std::shared_mutex mtx;
    std::unordered_set<std::string> ingested;
    std::deque<ObservabilityEventMetaEntry> log;  // append-only; the listener keeps pointers into it
    std::unordered_map<uint32_t, uint32_t> id_to_log_idx;
    ObservabilityEventMetaRegistry::Listener listener;
    uint64_t collisions = 0;
    uint64_t foreign_sections = 0;
    bool collision_logged = false;
};

State& state() {
    static ttsl::Indestructible<State> s;
    return s.get();
}

// Resolve a device VMA in .tt_observability_event_str to a string inside the mapped section, rebased by the section's
// own address so a non-ALLOC orphan at sh_addr 0 works too.
const char* resolve(std::span<std::byte> str_bytes, uint64_t str_vma, uint32_t ptr) {
    if (ptr < str_vma) {
        return nullptr;
    }
    const uint64_t off = static_cast<uint64_t>(ptr) - str_vma;
    if (off >= str_bytes.size()) {
        return nullptr;
    }
    const char* base = reinterpret_cast<const char*>(str_bytes.data());
    const size_t avail = str_bytes.size() - off;
    if (::strnlen(base + off, avail) == avail) {
        return nullptr;  // no NUL within the section
    }
    return base + off;
}

}  // namespace

ObservabilityEventMetaRegistry& ObservabilityEventMetaRegistry::instance() {
    static ObservabilityEventMetaRegistry inst;
    return inst;
}

void ObservabilityEventMetaRegistry::ingest_elf(const ll_api::ElfFile& elf, const std::string& elf_path) {
    State& s = state();
    {
        std::shared_lock rd(s.mtx);
        if (s.ingested.contains(elf_path)) {
            return;
        }
    }

    std::vector<ObservabilityEventMetaEntry> parsed;
    bool skipped_foreign = false;
    try {
        uint64_t meta_vma = 0;
        auto meta = elf.GetSectionContents(".tt_observability_event_meta", meta_vma);
        if (meta.empty()) {
            return;  // not a streaming-profiler build: nothing to register
        }
        uint64_t str_vma = 0;
        auto strs = elf.GetSectionContents(".tt_observability_event_str", str_vma);
        // The JIT cache key does not cover this section's layout, so a stale root can be reused and walking it at
        // our stride would bind plausible ids to wrong names; either guard failing means the section is not ours.
        if (strs.empty()) {
            log_debug(
                tt::LogLLRuntime,
                "observability-event-meta: '{}' has .tt_observability_event_meta but no .tt_observability_event_str -- "
                "foreign/stale record "
                "layout, "
                "ignoring the section (its events will render as Zone_<id>)",
                elf_path);
            skipped_foreign = true;
        } else if (meta.size() % sizeof(ObservabilityEventMetaRecord) != 0) {
            log_warning(
                tt::LogLLRuntime,
                "observability-event-meta: '{}' has a .tt_observability_event_meta of {} bytes, not a multiple of the "
                "{}-byte record "
                "stride -- foreign/stale record layout, ignoring the section",
                elf_path,
                meta.size(),
                sizeof(ObservabilityEventMetaRecord));
            skipped_foreign = true;
        }
        const size_t n = skipped_foreign ? 0 : meta.size() / sizeof(ObservabilityEventMetaRecord);
        parsed.reserve(n);
        for (size_t i = 0; i < n; i++) {
            ObservabilityEventMetaRecord rec{};
            std::memcpy(&rec, meta.data() + i * sizeof(ObservabilityEventMetaRecord), sizeof(rec));
            const char* name = resolve(strs, str_vma, rec.name_ptr);
            const char* file = resolve(strs, str_vma, rec.file_ptr);
            if (name == nullptr) {
                continue;
            }
            parsed.push_back(ObservabilityEventMetaEntry{
                rec.event_id & TT_OBSERVABILITY_EVENT_ID_MASK, name, file != nullptr ? file : "", rec.line});
        }
    } catch (const std::exception& e) {
        // Non-fatal: a kernel whose events cannot be named still profiles, rendering as "Zone_<id>".
        log_debug(tt::LogLLRuntime, "observability-event-meta: could not read '{}': {}", elf_path, e.what());
    }

    std::unique_lock wr(s.mtx);
    if (!s.ingested.insert(elf_path).second) {
        return;
    }
    if (skipped_foreign) {
        s.foreign_sections++;
    }
    std::vector<const ObservabilityEventMetaEntry*> added;
    for (auto& e : parsed) {
        auto it = s.id_to_log_idx.find(e.event_id);
        if (it != s.id_to_log_idx.end()) {
            const ObservabilityEventMetaEntry& prev = s.log[it->second];
            if (prev.name != e.name || prev.file != e.file || prev.line != e.line) {
                s.collisions++;
                if (!s.collision_logged) {
                    s.collision_logged = true;
                    log_warning(
                        tt::LogLLRuntime,
                        "observability-event-meta: structural event id {} (tu {} local {}) claimed by two source "
                        "locations: "
                        "'{}' ({}:{}) and '{}' ({}:{}). Event names for these will be wrong. This means two "
                        "translation units share a tu_id -- see get_or_assign_profiler_tu_id in "
                        "jit_build/build.cpp.",
                        e.event_id,
                        TT_OBSERVABILITY_EVENT_TU_OF(e.event_id),
                        TT_OBSERVABILITY_EVENT_LOCAL_OF(e.event_id),
                        prev.name,
                        prev.file,
                        prev.line,
                        e.name,
                        e.file,
                        e.line);
                }
            }
            continue;  // first writer wins: a name never changes under a consumer that already read it
        }
        s.id_to_log_idx.emplace(e.event_id, static_cast<uint32_t>(s.log.size()));
        added.push_back(&s.log.emplace_back(std::move(e)));
    }
    if (s.listener && !added.empty()) {
        s.listener(added);
    }
}

void ObservabilityEventMetaRegistry::set_listener(Listener listener) {
    State& s = state();
    std::unique_lock wr(s.mtx);
    std::vector<const ObservabilityEventMetaEntry*> all;
    all.reserve(s.log.size());
    for (const ObservabilityEventMetaEntry& e : s.log) {
        all.push_back(&e);
    }
    if (!all.empty()) {
        listener(all);
    }
    s.listener = std::move(listener);
}

uint64_t ObservabilityEventMetaRegistry::collisions() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.collisions;
}

uint64_t ObservabilityEventMetaRegistry::foreign_sections() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.foreign_sections;
}

}  // namespace tt::llrt
