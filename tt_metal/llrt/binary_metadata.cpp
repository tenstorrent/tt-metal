// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_metadata.hpp"

#include <cstddef>
#include <cstring>

#include <tt-logger/tt-logger.hpp>

#include "tt_elffile.hpp"

namespace ll_api {

namespace {

BufRwInfo parse_buf_rw(const ElfFile& elf) {
    BufRwInfo info;
    uint64_t vma = 0;
    auto bytes = elf.GetSectionContents(kBufRwSectionName, vma);
    if (bytes.empty()) {
        return info;
    }
    if (bytes.size() % sizeof(BufRwRecord) != 0) {
        // Not our fixed 8-byte stride -> foreign/stale section; treat as un-analyzable rather than
        // misparse a partial record into a wrong (slot, kind).
        log_warning(
            tt::LogLLRuntime,
            "buf_rw: .tt.BUF_RW is {} bytes, not a multiple of the {}-byte record stride -- ignoring as opaque",
            bytes.size(),
            sizeof(BufRwRecord));
        info.opaque = true;
        return info;
    }
    const size_t n = bytes.size() / sizeof(BufRwRecord);
    for (size_t i = 0; i < n; ++i) {
        BufRwRecord rec{};
        std::memcpy(&rec, bytes.data() + i * sizeof(BufRwRecord), sizeof(rec));
        switch (static_cast<BufRwKind>(rec.kind)) {
            case BufRwKind::Read: info.reads.insert(rec.slot); break;
            case BufRwKind::Write: info.writes.insert(rec.slot); break;
            case BufRwKind::Opaque:
            default:
                // OPAQUE (0) or any unrecognized kind -> conservative bail. Making 0 the opaque value
                // means a zero-filled/truncated record can never masquerade as a plain read.
                info.opaque = true;
                break;
        }
    }
    return info;
}

}  // namespace

BinaryMetadata parse_binary_metadata(const ElfFile& elf) {
    BinaryMetadata md;
    md.buf_rw = parse_buf_rw(elf);
    // Future (separate commit): fold ZoneMetaRegistry's .tt_zone_meta / .tt_zone_str harvest in here so it
    // shares this same single ELF open instead of re-reading the file in llrt::get_risc_binary.
    return md;
}

}  // namespace ll_api
