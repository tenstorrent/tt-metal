// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zone_meta.hpp"

#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>

#include "hostdevcommon/profiler_zone_id.h"
#include "tt_elffile.hpp"

namespace tt::llrt {

namespace {

// The on-wire record, mirroring TT_ZONE_DEFINE_ID in hostdevcommon/profiler_zone_id.h. The host walks the
// section at a fixed 16-byte stride and does no parsing, so these records must stay fixed-length.
struct ZoneMetaRecord {
    uint32_t zone_id;
    uint32_t name_ptr;  // VMA into .tt_zone_str
    uint32_t file_ptr;  // VMA into .tt_zone_str
    uint32_t line;
};
static_assert(
    sizeof(ZoneMetaRecord) == TT_ZONE_META_RECORD_BYTES, "the device emitter writes exactly four .long fields");

struct State {
    mutable std::shared_mutex mtx;
    std::unordered_set<std::string> ingested;
    std::vector<ZoneMetaEntry> log;  // append-only; the consumer's delta source
    std::unordered_map<uint32_t, uint32_t> id_to_log_idx;
    uint64_t collisions = 0;
    uint64_t records = 0;
    uint64_t foreign_sections = 0;
    bool collision_logged = false;
};

State& state() {
    static State s;
    return s;
}

// Resolve a device VMA in .tt_zone_str to a NUL-terminated string inside the mapped section bytes, or
// nullptr if it lands outside the section or is unterminated. Rebasing by the section's own address (as
// the DPRINT string table does) also covers the linker leaving it a non-ALLOC orphan at sh_addr 0.
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

void ZoneMetaRegistry::ingest_elf(const std::string& elf_path) {
    State& s = state();
    {
        std::shared_lock rd(s.mtx);
        if (s.ingested.count(elf_path) != 0) {
            return;
        }
    }

    std::vector<ZoneMetaEntry> parsed;
    bool skipped_foreign = false;
    try {
        ll_api::ElfFile elf;
        elf.ReadImage(elf_path);
        uint64_t meta_vma = 0;
        auto meta = elf.GetSectionContents(".tt_zone_meta", meta_vma);
        if (!meta.empty()) {
            uint64_t str_vma = 0;
            auto strs = elf.GetSectionContents(".tt_zone_str", str_vma);
            // The JIT cache key does not cover this section's layout, so a root written by an older
            // layout can still be reused, and walking one at our stride would mint plausible ids bound to
            // the wrong names. Our emitter always writes both sections and every record is 16 bytes, so
            // either guard failing means the section is not ours; reject it whole.
            if (strs.empty()) {
                log_debug(
                    tt::LogLLRuntime,
                    "zone-meta: '{}' has .tt_zone_meta but no .tt_zone_str -- foreign/stale record layout, "
                    "ignoring the section (its zones will render as Zone_<id>)",
                    elf_path);
                skipped_foreign = true;
            } else if (meta.size() % sizeof(ZoneMetaRecord) != 0) {
                log_warning(
                    tt::LogLLRuntime,
                    "zone-meta: '{}' has a .tt_zone_meta of {} bytes, not a multiple of the {}-byte record "
                    "stride -- foreign/stale record layout, ignoring the section",
                    elf_path,
                    meta.size(),
                    sizeof(ZoneMetaRecord));
                skipped_foreign = true;
            }
            const size_t n = skipped_foreign ? 0 : meta.size() / sizeof(ZoneMetaRecord);
            parsed.reserve(n);
            for (size_t i = 0; i < n; i++) {
                ZoneMetaRecord rec{};
                std::memcpy(&rec, meta.data() + i * sizeof(ZoneMetaRecord), sizeof(rec));
                const char* name = resolve(strs, str_vma, rec.name_ptr);
                const char* file = resolve(strs, str_vma, rec.file_ptr);
                if (name == nullptr) {
                    continue;
                }
                parsed.push_back(
                    ZoneMetaEntry{rec.zone_id & TT_ZONE_ID_MASK, name, file != nullptr ? file : "", rec.line});
            }
        }
    } catch (const std::exception& e) {
        // Non-fatal: a kernel whose zones cannot be named still profiles, rendering as "Zone_<id>".
        log_debug(tt::LogLLRuntime, "zone-meta: could not read '{}': {}", elf_path, e.what());
    }

    std::unique_lock wr(s.mtx);
    if (!s.ingested.insert(elf_path).second) {
        return;
    }
    if (skipped_foreign) {
        s.foreign_sections++;
    }
    for (auto& e : parsed) {
        s.records++;
        auto it = s.id_to_log_idx.find(e.zone_id);
        if (it != s.id_to_log_idx.end()) {
            const ZoneMetaEntry& prev = s.log[it->second];
            if (prev.name != e.name || prev.file != e.file || prev.line != e.line) {
                s.collisions++;
                if (!s.collision_logged) {
                    s.collision_logged = true;
                    log_warning(
                        tt::LogLLRuntime,
                        "zone-meta: structural zone id {} (tu {} local {}) claimed by two source locations: "
                        "'{}' ({}:{}) and '{}' ({}:{}). Zone names for these will be wrong. This means two "
                        "translation units share a tu_id -- see get_or_assign_profiler_tu_id in "
                        "jit_build/build.cpp.",
                        e.zone_id,
                        TT_ZONE_TU_OF(e.zone_id),
                        TT_ZONE_LOCAL_OF(e.zone_id),
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
        s.id_to_log_idx.emplace(e.zone_id, static_cast<uint32_t>(s.log.size()));
        s.log.push_back(std::move(e));
    }
}

uint32_t ZoneMetaRegistry::additions_since(uint32_t from, std::vector<ZoneMetaEntry>& out) const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    const uint32_t end = static_cast<uint32_t>(s.log.size());
    for (uint32_t i = from; i < end; i++) {
        out.push_back(s.log[i]);
    }
    return end;
}

uint64_t ZoneMetaRegistry::collisions() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return s.collisions;
}

ZoneMetaRegistry::Stats ZoneMetaRegistry::stats() const {
    const State& s = state();
    std::shared_lock rd(s.mtx);
    return Stats{static_cast<uint64_t>(s.ingested.size()), s.records, s.foreign_sections};
}

}  // namespace tt::llrt
