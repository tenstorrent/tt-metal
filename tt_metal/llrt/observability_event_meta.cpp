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
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/observability_event_id.h"
#include "tt_elffile.hpp"

namespace tt::llrt {

namespace {

// The fixed part of a .tt_observability_event_meta record; the site's metadata fields follow
// (hostdev/observability_event_meta.h).
struct RecordHeader {
    uint32_t event_id;       // the handle's VMA in .tt_observability_event_ids, which RebaseZoneIds has already moved
    uint32_t signature_ptr;  // VMA into .tt_observability_event_str: "<type name>:<field codes>"
    uint32_t file_ptr;       // VMA into .tt_observability_event_str
    uint32_t line;
};
static_assert(sizeof(RecordHeader) == tt::debug_event::kRecordHeaderBytes);

constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct State {
    mutable std::shared_mutex mtx;
    std::unordered_map<std::string, uint32_t> base_by_path;  // every image seen, and where its ids start
    uint32_t next_id = 0;
    std::deque<ObservabilityEventMetaEntry> log;  // append-only; the listener keeps pointers into it
    std::vector<int32_t> log_idx_by_id = std::vector<int32_t>(TT_OBSERVABILITY_EVENT_ID_COUNT, -1);
    ObservabilityEventMetaRegistry::Listener listener;
    std::unordered_set<std::string> strings;  // interned metadata strings; node-based, so pointers stay valid
    uint64_t malformed_records = 0;
    uint64_t foreign_sections = 0;
    bool malformed_logged = false;
};

State& state() {
    static ttsl::Indestructible<State> s;
    return s.get();
}

// Resolve a device VMA in .tt_observability_event_str to a string inside the mapped section, rebased by the section's
// own address so a non-ALLOC orphan at sh_addr 0 works too.
const char* resolve(std::span<const std::byte> str_bytes, uint64_t str_vma, uint32_t ptr) {
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

// The field codes after the signature's last ':', with a nested struct's braces balanced; nullptr if malformed.
const char* field_codes(const char* sig) {
    const char* colon = sig != nullptr ? std::strrchr(sig, ':') : nullptr;
    if (colon == nullptr) {
        return nullptr;
    }
    int depth = 0;
    for (const char* c = colon + 1; *c != 0; c++) {
        depth += *c == '{' ? 1 : *c == '}' ? -1 : 0;
        if (depth < 0) {
            return nullptr;
        }
    }
    return depth == 0 ? colon + 1 : nullptr;
}

}  // namespace

ParsedMetaRecords parse_meta_records(
    std::span<const std::byte> meta, std::span<const std::byte> str, uint64_t str_vma) {
    ParsedMetaRecords out;
    size_t off = 0;
    while (off + sizeof(RecordHeader) <= meta.size()) {
        RecordHeader h{};
        std::memcpy(&h, meta.data() + off, sizeof(h));
        const char* sig = resolve(str, str_vma, h.signature_ptr);
        const char* codes = field_codes(sig);
        if (codes == nullptr) {
            out.stopped_early = true;  // cannot size this record, so nothing after it can be found either
            return out;
        }
        ObservabilityEventMetaEntry e{.event_id = h.event_id, .line = h.line};
        const char* file = resolve(str, str_vma, h.file_ptr);
        e.file = file != nullptr ? file : "";
        e.meta.signature = sig;
        size_t pos = off + sizeof(RecordHeader);
        for (const char* c = codes; *c != 0; c++) {
            if (*c == '{' || *c == '}') {
                continue;  // a nested struct's bounds; its fields are laid out in place
            }
            const size_t bytes = tt::debug_event::detail::code_bytes(*c);
            if (bytes == 0) {
                out.stopped_early = true;  // a field code this reader does not know
                return out;
            }
            pos = align_up(pos, bytes < 4 ? bytes : 4);
            if (pos + bytes > meta.size()) {
                out.stopped_early = true;
                return out;
            }
            tt::debug_event::FieldValue v{.code = *c};
            std::memcpy(&v.bits, meta.data() + pos, bytes);  // little-endian host and device
            if (*c == 's') {
                v.s = resolve(str, str_vma, static_cast<uint32_t>(v.bits));
                if (v.s == nullptr) {
                    out.stopped_early = true;
                    return out;
                }
                if (e.name.empty()) {
                    e.name = v.s;
                }
            }
            e.meta.fields.push_back(v);
            pos += bytes;
        }
        off = align_up(pos, 4);
        out.entries.push_back(std::move(e));
    }
    return out;
}

ObservabilityEventMetaRegistry& ObservabilityEventMetaRegistry::instance() {
    static ObservabilityEventMetaRegistry inst;
    return inst;
}

void ObservabilityEventMetaRegistry::ingest_elf(const std::string& elf_path, ll_api::ElfFile& elf) {
    uint64_t ids_vma = 0;
    const size_t count = elf.GetSectionContents(".tt_observability_event_ids", ids_vma).size();
    if (count == 0) {
        return;  // no zone sites: dispatch kernels, the relay, anything built without the producer
    }

    State& s = state();
    uint32_t base = 0;
    bool first = false;
    {
        std::unique_lock wr(s.mtx);
        auto [it, inserted] = s.base_by_path.try_emplace(elf_path, s.next_id);
        if (inserted) {
            // The block must stay below the stall id, the one value the device emits without a record.
            TT_FATAL(
                static_cast<uint64_t>(s.next_id) + count <= TT_ZONE_STALL_ID,
                "streaming profiler: loading '{}' ({} zone sites) would exceed the {}-id zone space ({} assigned); "
                "this process has loaded more distinct device zone sites than the 16-bit id supports",
                elf_path,
                count,
                TT_ZONE_STALL_ID,
                s.next_id);
            s.next_id += static_cast<uint32_t>(count);
            first = true;
        }
        base = it->second;
    }
    // The image is this caller's; its text is packed for the device after we return, so the rebase must
    // land here whether or not this path's names were already registered.
    elf.RebaseZoneIds(base);
    if (!first) {
        return;
    }

    // Parsed outside the lock; string fields still point into the ELF until they are interned below.
    std::vector<ObservabilityEventMetaEntry> parsed;
    bool skipped_foreign = false;
    uint64_t malformed = 0;
    try {
        uint64_t meta_vma = 0;
        auto meta = elf.GetSectionContents(".tt_observability_event_meta", meta_vma);
        uint64_t str_vma = 0;
        auto strs = elf.GetSectionContents(".tt_observability_event_str", str_vma);
        // The JIT cache key does not cover this section's layout, so a stale root can be reused; every record is
        // checked against its own signature, and the walk stops at the first one it cannot size.
        if (meta.empty() || strs.empty()) {
            log_debug(
                tt::LogLLRuntime,
                "zone-meta: '{}' has zone ids but no .tt_observability_event_meta/.tt_observability_event_str -- "
                "foreign/stale record layout, "
                "ignoring the section (its zones will render as Zone_<id>)",
                elf_path);
            skipped_foreign = true;
        }
        if (!skipped_foreign) {
            ParsedMetaRecords records = parse_meta_records(meta, strs, str_vma);
            skipped_foreign = records.stopped_early;
            for (ObservabilityEventMetaEntry& e : records.entries) {
                // Every record's id is a handle in this image's block; anything else is a layout we do not
                // understand.
                if (e.name.empty() || e.event_id < base || e.event_id >= base + count) {
                    malformed++;
                    continue;
                }
                parsed.push_back(std::move(e));
            }
        }
    } catch (const std::exception& e) {
        // Non-fatal: a kernel whose events cannot be named still profiles, rendering as "Zone_<id>".
        log_debug(tt::LogLLRuntime, "observability-event-meta: could not read '{}': {}", elf_path, e.what());
    }

    std::unique_lock wr(s.mtx);
    if (skipped_foreign) {
        s.foreign_sections++;
    }
    std::vector<const ObservabilityEventMetaEntry*> added;
    for (auto& e : parsed) {
        if (s.log_idx_by_id[e.event_id] >= 0) {
            malformed++;  // two records on one handle: not something the emitter can produce
            continue;
        }
        // The ELF is gone once its image is packed; the registry owns every string a site's metadata names.
        e.meta.signature = *s.strings.emplace(e.meta.signature).first;
        for (tt::debug_event::FieldValue& v : e.meta.fields) {
            if (v.s != nullptr) {
                v.s = s.strings.emplace(v.s).first->c_str();
            }
        }
        s.log_idx_by_id[e.event_id] = static_cast<int32_t>(s.log.size());
        added.push_back(&s.log.emplace_back(std::move(e)));
    }
    if (malformed != 0) {
        s.malformed_records += malformed;
        if (!s.malformed_logged) {
            s.malformed_logged = true;
            log_warning(
                tt::LogLLRuntime,
                "zone-meta: '{}' has {} zone record(s) whose id is outside its own block [{}, {}) or unnamed; those "
                "zones will render as Zone_<id> (stale .tt_observability_event_meta layout in the JIT cache?)",
                elf_path,
                malformed,
                base,
                base + count);
        }
    }
    log_debug(
        tt::LogLLRuntime, "zone-meta: '{}' -> zone ids [{}, {}), {} named", elf_path, base, base + count, added.size());
    for (const ObservabilityEventMetaEntry* e : added) {
        if (auto c = e->meta.as<tt::debug_event::ZoneColorMeta>()) {
            log_debug(
                tt::LogLLRuntime,
                "zone-meta: id {} '{}' ({}:{}) carries {} {{color={:#08x}}}",
                e->event_id,
                c->name,
                e->file,
                e->line,
                e->meta.type_name(),
                c->color);
        }
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

uint32_t ObservabilityEventMetaRegistry::ids_assigned() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.next_id;
}

uint64_t ObservabilityEventMetaRegistry::malformed_records() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.malformed_records;
}

uint64_t ObservabilityEventMetaRegistry::foreign_sections() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.foreign_sections;
}

}  // namespace tt::llrt
