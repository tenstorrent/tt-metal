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

// The on-wire record, mirroring TT_ZONE_DEFINE_ID in hostdevcommon/profiler_zone_id.h. FOUR u32s,
// SIXTEEN BYTES, and the host walks the section with THAT as its stride -- no parsing, no per-record
// length, nothing to round up to an alignment. That matters: a variable-length record (id + inline
// NUL-terminated string) has to round each advance up to the section's alignment by hand, and getting
// that wrong by one does not fail, it DESYNCHRONISES -- reading the tail of one record as the head of the
// next and minting plausible-looking ids bound to the wrong names. A fixed stride makes that class of bug
// unrepresentable.
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
    std::unordered_set<std::string> ingested;              // ELF paths already read
    std::vector<ZoneMetaEntry> log;                        // append-only; the consumer's delta source
    std::unordered_map<uint32_t, uint32_t> id_to_log_idx;  // dedup + collision detection
    uint64_t collisions = 0;
    uint64_t records = 0;
    uint64_t foreign_sections = 0;  // ELFs whose .tt_zone_meta failed a format guard
    bool collision_logged = false;
};

State& state() {
    static State s;
    return s;
}

// Resolve a device VMA in .tt_zone_str to a NUL-terminated string inside the mapped section bytes.
// Returns nullptr if the pointer lands outside the section or the string is unterminated -- the same
// rebase trick the DPRINT string table uses: file_bytes + (device_ptr - sh_addr). This also handles the
// case where the linker script does not mention the section and it lands as a non-ALLOC orphan at
// sh_addr 0: the rebase is by the section's OWN address, so 0 is just another base.
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
        return nullptr;  // unterminated: refuse rather than run off the end
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
            // ---- Format guards. Both of these REJECT the section whole rather than parse part of it.
            //
            // They exist because the JIT cache is keyed on the build recipe, not on this section's layout,
            // so a cache root written by a DIFFERENT layout of .tt_zone_meta can still be sitting on disk
            // and be reused. Parsing one of those as a fixed-stride array is exactly the failure mode a
            // stride is supposed to prevent -- it would not error, it would read the middle of one record
            // as the head of the next and mint plausible-looking ids bound to the wrong names.
            //
            // Guard 1: our emitter ALWAYS writes both sections, so .tt_zone_meta without .tt_zone_str is
            // by definition not our format.
            // Guard 2: every record is exactly 16 bytes, so our size is always a multiple of 16. A
            // remainder means the records are not ours.
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
                    continue;  // a record we cannot name is worth nothing; skip it, keep the rest
                }
                parsed.push_back(
                    ZoneMetaEntry{rec.zone_id & TT_ZONE_ID_MASK, name, file != nullptr ? file : "", rec.line});
            }
        }
    } catch (const std::exception& e) {
        // Deliberately non-fatal. A kernel whose zones cannot be named still profiles correctly; the
        // consumer falls back to "Zone_<id>" for it, and the run continues.
        log_debug(tt::LogLLRuntime, "zone-meta: could not read '{}': {}", elf_path, e.what());
    }

    std::unique_lock wr(s.mtx);
    if (!s.ingested.insert(elf_path).second) {
        return;  // another thread got here first
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
            continue;  // first writer wins, so a name never changes under a consumer that already saw it
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
