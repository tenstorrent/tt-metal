// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop SHM schema — shared between collector (writer) and viewer (reader).
//
// One file per chip at /dev/shm/tt_device_<asic_id>_util, mirroring the
// naming pattern of tt_metal's memory_stats_shm.hpp so other tools
// (tt-mgmt, exporters) can find it without coordination.
//
// Phase 1 (this commit) populates only `dispatched` and `dispatch_busy_p1000`.
// Phase 2 (on-chip sampler) will populate the compute/unpack/pack/stall/NOC
// fields via the mechanism described in tt_metal/tools/ttnvtop/PLAN.md §5.
// The viewer must handle either case: if `signal_sources` has the COMPUTE
// bit set, render compute%; otherwise fall back to dispatch%.

#pragma once

#include <cstdint>

namespace ttnvtop {

// Bump on any binary-incompatible change to the structs below.
constexpr uint16_t kShmVersion = 1;

constexpr char kShmMagic[4] = {'T', 'T', 'U', 'T'};

// Bitfield for UtilShmHeader::signal_sources.
enum SignalSource : uint32_t {
    SIGNAL_SRC_NONE = 0,
    SIGNAL_SRC_DISPATCH = 1u << 0,  // go_msg.signal sampling (Phase 1)
    SIGNAL_SRC_COMPUTE = 1u << 1,   // perf-counter sampling (Phase 2+)
};

// Written once at collector startup; `last_update_us` refreshed every tick.
struct UtilShmHeader {
    char magic[4];              // must be kShmMagic ('TTUT')
    uint16_t version;           // kShmVersion
    uint16_t struct_size;       // sizeof(PerCoreView), for forward compat
    uint64_t asic_id;           // chip unique id from UMD
    uint32_t arch_id;           // tt::ARCH as u32
    uint32_t signal_sources;    // bitmask of SignalSource
    uint64_t epoch_us;          // CLOCK_MONOTONIC microseconds at collector start
    uint64_t last_update_us;    // CLOCK_MONOTONIC at last write
    uint32_t num_cores;         // number of PerCoreView entries that follow
    uint32_t host_assigned_id;  // current program id, 0 if unknown
    uint32_t collector_pid;
    // Live AICLK frequency in MHz, refreshed by the collector on every
    // publish tick via TTDevice::get_clock(). 0 means unknown — viewer
    // should then omit throughput rendering.
    uint32_t aiclk_mhz;
    uint32_t reserved[4];
};
static_assert(sizeof(UtilShmHeader) == 72, "UtilShmHeader must be 72 bytes");

// One per worker core, appended after the header.
// Rates are encoded as per-mille (0..1000) so the struct stays u16-sized and
// aligned; viewers divide by 10 to get "%" with one decimal.
struct PerCoreView {
    uint8_t noc_x;
    uint8_t noc_y;
    uint8_t logical_x;
    uint8_t logical_y;
    uint8_t is_remote;
    uint8_t dispatched;  // latest 1-bit from go_msg.signal (RUN_MSG_GO?)
    uint16_t reserved_0;
    uint16_t dispatch_busy_p1000;  // Phase 1 rolling dispatch-occupancy
    uint16_t compute_busy_p1000;   // Phase 2+: FPU/MATH busy
    uint16_t unpack_busy_p1000;    // Phase 2+
    uint16_t pack_busy_p1000;      // Phase 2+
    uint16_t stall_p1000;          // Phase 2+
    uint16_t noc0_in_mbps;
    uint16_t noc0_out_mbps;
    uint16_t noc1_in_mbps;
    uint16_t noc1_out_mbps;
    uint32_t samples_seen;
    uint32_t last_kernel_id;
    uint32_t reserved_1;
};
static_assert(sizeof(PerCoreView) == 40, "PerCoreView must be 40 bytes");

constexpr size_t shm_file_size(size_t num_cores) { return sizeof(UtilShmHeader) + num_cores * sizeof(PerCoreView); }

}  // namespace ttnvtop
