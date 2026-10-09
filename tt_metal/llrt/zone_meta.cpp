// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zone_meta.hpp"

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

#include "hostdev/profiler_zone_id.h"
#include "tt_elffile.hpp"

namespace tt::llrt {

namespace {

// The fixed part of a .tt_zone_meta record; the site's metadata fields follow (hostdev/debug_event_meta.h).
struct RecordHeader {
    uint32_t zone_id;        // the handle's VMA in .tt_zone_ids, which RebaseZoneIds has already moved
    uint32_t signature_ptr;  // VMA into .tt_zone_str: "<type name>:<field codes>"
    uint32_t file_ptr;       // VMA into .tt_zone_str
    uint32_t line;
};
static_assert(sizeof(RecordHeader) == tt::debug_event::kRecordHeaderBytes);

constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct State {
    mutable std::shared_mutex mtx;
    std::unordered_map<std::string, uint32_t> base_by_path;  // every image seen, and where its ids start
    uint32_t next_id = 0;
    std::deque<ZoneMetaEntry> log;  // append-only; the listener keeps pointers into it
    std::vector<int32_t> log_idx_by_id = std::vector<int32_t>(TT_ZONE_ID_COUNT, -1);
    ZoneMetaRegistry::Listener listener;
    std::unordered_set<std::string> strings;  // interned metadata strings; node-based, so pointers stay valid
    uint64_t malformed_records = 0;
    uint64_t foreign_sections = 0;
    bool malformed_logged = false;
};

State& state() {
    static ttsl::Indestructible<State> s;
    return s.get();
}

// Resolve a device VMA in .tt_zone_str to a string inside the mapped section, rebased by the section's own
// address so a non-ALLOC orphan at sh_addr 0 works too.
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

ZoneMetaRegistry& ZoneMetaRegistry::instance() {
    static ZoneMetaRegistry inst;
    return inst;
}

void ZoneMetaRegistry::ingest_elf(const std::string& elf_path, ll_api::ElfFile& elf) {
    uint64_t ids_vma = 0;
    const size_t count = elf.GetSectionContents(".tt_zone_ids", ids_vma).size();
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
    std::vector<ZoneMetaEntry> parsed;
    bool skipped_foreign = false;
    uint64_t malformed = 0;
    try {
        uint64_t meta_vma = 0;
        auto meta = elf.GetSectionContents(".tt_zone_meta", meta_vma);
        uint64_t str_vma = 0;
        auto strs = elf.GetSectionContents(".tt_zone_str", str_vma);
        // The JIT cache key does not cover this section's layout, so a stale root can be reused; every record is
        // checked against its own signature, and the walk stops at the first one it cannot size.
        if (meta.empty() || strs.empty()) {
            log_debug(
                tt::LogLLRuntime,
                "zone-meta: '{}' has zone ids but no .tt_zone_meta/.tt_zone_str -- foreign/stale record layout, "
                "ignoring the section (its zones will render as Zone_<id>)",
                elf_path);
            skipped_foreign = true;
        }
        size_t off = 0;
        while (!skipped_foreign && off + sizeof(RecordHeader) <= meta.size()) {
            RecordHeader h{};
            std::memcpy(&h, meta.data() + off, sizeof(h));
            const char* sig = resolve(strs, str_vma, h.signature_ptr);
            const char* colon = sig != nullptr ? std::strrchr(sig, ':') : nullptr;
            if (colon == nullptr) {
                skipped_foreign = true;  // cannot size this record, so nothing after it can be found either
                break;
            }
            ZoneMetaEntry e{.zone_id = h.zone_id, .line = h.line};
            const char* file = resolve(strs, str_vma, h.file_ptr);
            e.file = file != nullptr ? file : "";
            e.meta.signature = sig;
            size_t pos = off + sizeof(RecordHeader);
            bool ok = true;
            for (const char* c = colon + 1; *c != 0; c++) {
                if (*c == '{' || *c == '}') {
                    continue;  // a nested struct's bounds; its fields are laid out in place
                }
                const size_t bytes = tt::debug_event::detail::code_bytes(*c);
                pos = align_up(pos, bytes < 4 ? bytes : 4);
                if (bytes == 0 || pos + bytes > meta.size()) {
                    ok = false;
                    break;
                }
                tt::debug_event::FieldValue v{.code = *c};
                std::memcpy(&v.bits, meta.data() + pos, bytes);  // little-endian host and device
                if (*c == 's') {
                    v.s = resolve(strs, str_vma, static_cast<uint32_t>(v.bits));
                    ok = ok && v.s != nullptr;
                    if (e.name.empty() && v.s != nullptr) {
                        e.name = v.s;
                    }
                }
                e.meta.fields.push_back(v);
                pos += bytes;
            }
            if (!ok) {
                skipped_foreign = true;
                break;
            }
            off = align_up(pos, 4);
            // Every record's id is a handle in this image's block; anything else is a layout we do not understand.
            if (e.name.empty() || e.zone_id < base || e.zone_id >= base + count) {
                malformed++;
                continue;
            }
            parsed.push_back(std::move(e));
        }
    } catch (const std::exception& e) {
        // Non-fatal: a kernel whose zones cannot be named still profiles, rendering as "Zone_<id>".
        log_debug(tt::LogLLRuntime, "zone-meta: could not read '{}': {}", elf_path, e.what());
    }

    std::unique_lock wr(s.mtx);
    if (skipped_foreign) {
        s.foreign_sections++;
    }
    std::vector<const ZoneMetaEntry*> added;
    for (auto& e : parsed) {
        if (s.log_idx_by_id[e.zone_id] >= 0) {
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
        s.log_idx_by_id[e.zone_id] = static_cast<int32_t>(s.log.size());
        added.push_back(&s.log.emplace_back(std::move(e)));
    }
    if (malformed != 0) {
        s.malformed_records += malformed;
        if (!s.malformed_logged) {
            s.malformed_logged = true;
            log_warning(
                tt::LogLLRuntime,
                "zone-meta: '{}' has {} zone record(s) whose id is outside its own block [{}, {}) or unnamed; those "
                "zones will render as Zone_<id> (stale .tt_zone_meta layout in the JIT cache?)",
                elf_path,
                malformed,
                base,
                base + count);
        }
    }
    log_debug(
        tt::LogLLRuntime, "zone-meta: '{}' -> zone ids [{}, {}), {} named", elf_path, base, base + count, added.size());
    for (const ZoneMetaEntry* e : added) {
        if (auto c = e->meta.as<tt::debug_event::ZoneColorMeta>()) {
            log_debug(
                tt::LogLLRuntime,
                "zone-meta: id {} '{}' ({}:{}) carries {} {{color={:#08x}}}",
                e->zone_id,
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

void ZoneMetaRegistry::set_listener(Listener listener) {
    State& s = state();
    std::unique_lock wr(s.mtx);
    std::vector<const ZoneMetaEntry*> all;
    all.reserve(s.log.size());
    for (const ZoneMetaEntry& e : s.log) {
        all.push_back(&e);
    }
    if (!all.empty()) {
        listener(all);
    }
    s.listener = std::move(listener);
}

uint32_t ZoneMetaRegistry::ids_assigned() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.next_id;
}

uint64_t ZoneMetaRegistry::malformed_records() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.malformed_records;
}

uint64_t ZoneMetaRegistry::foreign_sections() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.foreign_sections;
}

}  // namespace tt::llrt
