// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop program registry — shared layout between the writer library
// (linked into tt-metal workloads when TTNVTOP_REGISTER_PROGRAMS=1) and
// the ttnvtop viewer (read-only consumer).
//
// The registry is a single file at /dev/shm/tt_program_registry.
// Writer side is a circular buffer: each program registration atomically
// claims the next slot via fetch_add on write_cursor, then memcpy's its
// entry into place. Reader side walks [0, write_cursor % capacity) and
// (write_cursor >= capacity ? [write_cursor % capacity, capacity) : empty)
// to see every live mapping.
//
// Layout is fixed-capacity (kRegCapacity entries) so the whole thing is a
// single mmap with no resize path. 16K entries × 112 B = ~1.8 MB — cheap.
//
// No tt-metal headers here: this file is included by the registrar library
// (which must stay dependency-free so it can be linked from tt-metal core
// without circular dependencies).

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ttnvtop {

constexpr char kRegistryMagic[4] = {'T', 'P', 'R', 'G'};
// v3 (Phase 2.1.c.i): adds RegistryEntry::cycles_total (monotonic per-program
// cycle accumulator since collector start). cycles_in_window kept for the
// live viewer's TIME% column; cycles_total is what compare.py audits against
// the device profiler. Old viewers that see version != kRegistryVersion
// gracefully refuse to map (existing behavior in viewer/main.cpp
// open_registry_shm).
constexpr uint16_t kRegistryVersion = 3;
constexpr uint32_t kRegistryCapacity = 16384;  // entries
constexpr uint32_t kRegistryNameMax = 96;      // bytes incl. null terminator

// Header at offset 0 of the SHM file. struct is plain-standard-layout
// (aside from the atomic, which on x86-64 is still trivially copyable).
struct RegistryHeader {
    char magic[4];                       // must equal kRegistryMagic
    uint16_t version;                    // kRegistryVersion
    uint16_t entry_size;                 // sizeof(RegistryEntry), forward-compat
    uint32_t capacity;                   // kRegistryCapacity
    uint32_t writer_pid;                 // pid of the workload process
    uint64_t epoch_us;                   // CLOCK_MONOTONIC at writer init
    std::atomic<uint32_t> write_cursor;  // total entries written (wraps at capacity)
    uint32_t reserved[4];
};
static_assert(sizeof(RegistryHeader) == 48, "RegistryHeader must be 48 bytes");

struct RegistryEntry {
    uint32_t runtime_id;          // program.get_runtime_id() (u16-ish)
    uint32_t pid;                 // writer pid — lets reader detect stale
    uint64_t epoch_us;            // when registered
    char name[kRegistryNameMax];  // null-terminated, truncated if too long
    // Phase 2.1.c: rolling-window cycle attribution. WRITTEN by the
    // ttnvtop-collector RingDrainThread; READ by the ttnvtop viewer for the
    // TIME% column. Workload writes runtime_id+pid+epoch_us+name (slot
    // claim); collector writes only this field (existing slot lookup). The
    // two fields are disjoint and updated independently — no atomic update
    // dance needed beyond a relaxed 64-bit store on x86-64.
    uint64_t cycles_in_window;
    // Phase 2.1.c.i: monotonic per-program cycle accumulator since collector
    // start. WRITTEN by the collector RingDrainThread (added each tick to
    // the existing kernel_cycles accumulator's `total`); READ by compare.py
    // for end-of-run audit. Disjoint from cycles_in_window: the latter
    // decays to 0 in the rolling window, this one never decreases.
    uint64_t cycles_total;
};
// Grew 120 -> 128 B (v3 schema). Costs ~128 KB extra per registry
// (16384 * 8 B). Version bump (2 -> 3) handled identically to v2: writer
// init zero-fills on mismatch, viewer refuses to map on mismatch.
static_assert(sizeof(RegistryEntry) == 128, "RegistryEntry must be 128 bytes (v3)");

constexpr size_t registry_file_size() {
    return sizeof(RegistryHeader) + static_cast<size_t>(kRegistryCapacity) * sizeof(RegistryEntry);
}

// Fixed path. Matches /dev/shm/tt_device_<id>_util convention.
constexpr const char* kRegistryShmPath = "/dev/shm/tt_program_registry";

// Env-var name used by the writer to gate initialization. When set to "1",
// the registrar library lazily mmaps the SHM file and begins publishing.
// When unset or any other value, the registrar is a hot-path no-op.
constexpr const char* kRegistryEnvVar = "TTNVTOP_REGISTER_PROGRAMS";

}  // namespace ttnvtop
