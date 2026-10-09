// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <set>
#include <string_view>

namespace ll_api {

class ElfFile;

// --- Op-to-op R/W inference (POC) -----------------------------------------------------------------
// The device-side NoC read/write APIs emit one fixed 8-byte record per analyzed access into the
// non-allocatable ".tt.BUF_RW" section: (object-slot, kind). The loader harvests it once, during the
// single ELF open that builds the device image (see llrt::get_binary_metadata), so the host can recover
// which bound objects a kernel reads vs writes -- the input a future trace-mode barrier-relaxation
// optimizer needs. Eventually the compiler emits these records; for now inline-asm annotations do (and,
// deliberately, the raw/un-analyzable NoC paths emit OPAQUE).
//
// OPAQUE is 0 so that a zero-filled / truncated / missing record degrades to the conservative
// "un-analyzable" state rather than being mistaken for a plain read.
enum class BufRwKind : uint32_t {
    Opaque = 0,  // raw/un-analyzable access: a detector must KEEP the barrier regardless of reads/writes
    Read = 1,    // reads the bound object at `slot`
    Write = 2,   // writes the bound object at `slot`
};
// The device emitter's on-ELF record: two little-endian u32s. `slot` identifies the bound object -- for
// now a tensor binding (KernelSpec tensor_bindings order), later possibly a buffer/shard/etc.; it is a
// sentinel (kBufRwOpaqueSlot) for OPAQUE.
struct BufRwRecord {
    uint32_t slot;
    uint32_t kind;
};
inline constexpr std::string_view kBufRwSectionName = ".tt.BUF_RW";
inline constexpr uint32_t kBufRwOpaqueSlot = 0xFFFFFFFFu;

// Decoded per-object R/W sets (each entry a bound-object slot). `opaque` set => at least one un-analyzable
// access was seen, so a detector must treat the binary conservatively (keep the op boundary) even if
// reads/writes look clean.
struct BufRwInfo {
    std::set<uint32_t> reads;
    std::set<uint32_t> writes;
    bool opaque = false;
};

// All host-side metadata harvested from one binary's ELF. Deliberately a bundle (not just BufRwInfo) so
// other per-ELF metadata gathered at load time -- e.g. profiler zone names (.tt_zone_meta), today
// re-opened separately by ZoneMetaRegistry -- can fold in here later and share the loader's one open.
struct BinaryMetadata {
    BufRwInfo buf_rw;
};

// Harvest all BinaryMetadata from an already-open ELF. This opens no file: the caller owns the ElfFile
// (the loader passes the same one it uses to build the device image), which is the whole point -- one
// open per binary. Must run before any in-place transform of the ELF (e.g. XIP) so it reads the pristine
// sections.
BinaryMetadata parse_binary_metadata(const ElfFile& elf);

}  // namespace ll_api
