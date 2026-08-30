// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop idle-eth aggregator journal (Phase 2.2, milestone M1).
//
// Sibling to util_sampler.h. Where util_sampler.h defines the PER-CORE ring
// that TRISC1 writes into Tensix L1, this header defines the PER-CHIP journal
// that an aggregator kernel on an idle ethernet core sweeps those rings into,
// and then PUSHES over fabric into an idle-eth L1 slot on the MMIO chip.
//
// Why a push and not a host pull: a host read of a remote chip is
// RemoteCommunicationLegacyFirmware::read_non_mmio, which writes a command
// into an ethernet core's firmware queue and polls for completion while
// holding UMD's NON_MMIO mutex. Under a workload that saturates ETH those
// cores do not service the command promptly and the host blocks for tens of
// seconds -- measured under Llama-3.3-70B. See PLAN_ETH_AGGREGATOR.md 3.3/5c.
// The host only ever reads the LANDING copy, on the MMIO chip, over PCIe.
//
// Single writer: exactly one aggregator kernel owns a given journal. The host
// is a pure reader and never writes into the journal region. `head` is written
// LAST on every update so a racing reader never sees a half-written entry --
// the same discipline util_sampler.h uses for its ring.

#pragma once

#include <cstdint>

// DELIBERATELY DEPENDENCY-FREE.
//
// util_sampler.h is a firmware-only header: it dereferences MEM_UTIL_SAMPLER_BASE
// and reads RISCV_DEBUG_REG_*, so it needs dev_mem_map.h, core_config.h and
// tensix.h on the include path. The host collector therefore does NOT include it
// -- it hand-mirrors util_sampler_entry_t (collector/main.cpp:88). This header is
// consumed by BOTH the aggregator kernel and the host, so it takes no firmware
// dependency and carries its own copy of the 16 B sample layout. The
// static_assert below ties the two together whenever both headers are visible,
// so the duplication cannot silently drift.

// Byte-identical to util_sampler_entry_t. Forwarded VERBATIM by the aggregator:
// no reinterpretation on the wire, so the host's existing delta/EWMA/kernel-
// attribution code needs no edit beyond the core_id demux.
struct util_agg_sample_t {
    uint32_t wall_clock_l;   // wraps every ~4.3 s at 1 GHz; host reconstructs a 64 b timeline
    uint32_t kernel_id;      // host_assigned_id at sample time. 0 = no program
    uint32_t fpu_count;      // PERF_CNT_OUT_H_FPU snapshot for whichever counter_sel is live
    uint8_t math_fidelity;   // 0 unset, 1 LoFi, 2 HiFi2, 4 HiFi4
    uint8_t counter_sel;     // 0 FPU_INSTRUCTION, 1 SFPU_INSTRUCTION
    uint8_t producer_riscv;  // 0 BRISC, 1 TRISC1
    uint8_t flags;           // bit 0: kernel-start marker
};
static_assert(sizeof(util_agg_sample_t) == 16);

#ifdef UTIL_SAMPLER_MAGIC
static_assert(
    sizeof(util_agg_sample_t) == sizeof(util_sampler_entry_t),
    "util_agg_sample_t has drifted from util_sampler_entry_t");
#endif

constexpr uint32_t UTIL_AGG_MAGIC = 0x47415454u;  // 'TTAG' little-endian
constexpr uint32_t UTIL_AGG_VERSION = 1u;

// Upper bound on Tensix cores an aggregator sweeps on one chip. Sized from the
// UNHARVESTED grids, because harvesting is not guaranteed:
//   WH  tensix_grid_size 8x10  =  80  (shipped parts harvest 2 rows -> 64, but 224
//                                      of 278 shipped descriptor entries carry
//                                      harvest_mask 0, so 64 is NOT a constant)
//   BH  TENSIX_GRID_SIZE 14x10 = 140  (a p150a measured here has 2 columns
//                                      harvested -> 120)
// 160 leaves headroom over the largest of those. This was 128, which is smaller
// than an unharvested Blackhole part. See PLAN_ETH_AGGREGATOR.md 2.1.
constexpr uint32_t UTIL_AGG_MAX_CORES = 160u;

// M1 uses a 32-BYTE entry rather than the 20 B the plan sketched.
//
// WH L1_ALIGNMENT is 16 B, so a fabric write into the landing journal must
// start 16 B-aligned. At 20 B per entry every other entry begins at an
// unaligned offset; at 24 B only even indices align. 32 B makes every entry
// index trivially aligned and makes the host decode a plain array index.
//
// The cost is 12 B/entry of ethernet traffic that carries nothing. At M1 rates
// (1 ms sampling, 64 cores => 64k entries/s) that is ~2 MB/s against a 100 Gb/s
// link -- irrelevant. If M2's 100 us sampling makes it matter, pack to 24 B and
// constrain writes to even entry indices; do not pack to 20 B.
// The 16 B sample is FIRST, not after the metadata. The aggregator NOC-reads a
// ring slot straight into this field, and a NOC read into L1 must be 16 B
// aligned on WH. With core_id/seq ahead of it the sample would land at
// entry_base+8 and every such read would be misaligned.
struct util_agg_entry_t {
    util_agg_sample_t sample;  // 16 B at offset 0, forwarded VERBATIM from the Tensix ring
    uint32_t core_id;          // index into the chip's Tensix core list (host-supplied ordering)
    uint32_t seq;              // per-core monotonic counter; gaps mean the aggregator dropped entries
    uint32_t reserved[2];      // pad to 32 B, see note above
};
static_assert(sizeof(util_agg_entry_t) == 32);
static_assert(sizeof(util_agg_entry_t) % 16 == 0, "entry must be a multiple of WH L1_ALIGNMENT");

// Header is 64 B so journal[] starts 16 B-aligned (and, incidentally, 64 B-aligned).
struct util_agg_msg_t {
    volatile uint32_t magic;         // 'TTAG'
    volatile uint32_t version;       // 1
    volatile uint32_t head;          // monotonic COUNT of entries ever written. Written LAST.
    volatile uint32_t capacity;      // entries in journal[]; head % capacity is the write slot
    volatile uint32_t num_cores;     // cores this aggregator sweeps. NEVER assume 64 (see 2.1)
    volatile uint32_t sweep_count;   // aggregator liveness heartbeat; host staleness check keys on this
    volatile uint32_t lost;          // entries dropped because a Tensix ring wrapped before we swept it
    volatile uint32_t src_chip;      // fabric node id of the chip this journal came from
    volatile uint32_t hdr_checksum;  // sum of the 8 u32s above, so a torn header is detectable
    volatile uint32_t reserved[7];   // pad to 64 B
    volatile util_agg_entry_t journal[];
};
static_assert(sizeof(util_agg_msg_t) == 64);
static_assert(sizeof(util_agg_msg_t) % 16 == 0, "journal[] must start 16 B-aligned");

// M1 journal sizing. 192 KiB / 32 B = 6144 entries.
//
// At M1's 1 ms sampling (64 cores => ~64k entries/s) that is ~96 ms of
// buffering, comfortably ahead of a 10 Hz host drain. At M2's 100 us it drops
// to ~9.6 ms and the host must drain at 50-100 Hz -- which is the drain rate
// M2 is specified around anyway. The buffer is not the M2 constraint; the
// drain cadence is.
constexpr uint32_t UTIL_AGG_JOURNAL_BYTES = 192u * 1024u;
constexpr uint32_t UTIL_AGG_CAPACITY = (UTIL_AGG_JOURNAL_BYTES - sizeof(util_agg_msg_t)) / sizeof(util_agg_entry_t);
static_assert(UTIL_AGG_CAPACITY == 6142);

// Header checksum over the 8 u32s preceding it. Cheap enough to recompute on
// every header push, and it is the ONLY thing that lets the host distinguish a
// torn header from a valid one -- there is no handshake with the aggregator.
inline uint32_t util_agg_hdr_checksum(
    uint32_t magic,
    uint32_t version,
    uint32_t head,
    uint32_t capacity,
    uint32_t num_cores,
    uint32_t sweep_count,
    uint32_t lost,
    uint32_t src_chip) {
    return magic + version + head + capacity + num_cores + sweep_count + lost + src_chip;
}

// PUSH ORDERING CONTRACT
//
// Each sweep that produces entries issues TWO fabric packets on the SAME
// connection, in this order:
//   1. the new journal[] entries (may be split in two if the ring wraps)
//   2. the 64 B header, with `head` advanced to cover them
//
// Packets on one fabric connection are delivered in order, so a host that sees
// an advanced `head` is guaranteed the entries behind it have landed. The
// checksum guards the header itself; `head`-written-last guards the entries.
// A host that sees a bad checksum must retry, NOT fall back to the previous
// head -- the entries are already committed and skipping them loses samples.

// Plain host-side view of the header.
//
// util_agg_msg_t's fields are volatile -- they are written by the aggregator and polled
// by a host that must not have them cached -- which makes the struct non-trivially
// copyable and therefore not a legal memcpy destination. This is the identical layout
// without the qualifier, for host code that reads a landed journal into a buffer.
struct util_agg_hdr_view_t {
    uint32_t magic;
    uint32_t version;
    uint32_t head;
    uint32_t capacity;
    uint32_t num_cores;
    uint32_t sweep_count;
    uint32_t lost;
    uint32_t src_chip;
    uint32_t hdr_checksum;
    uint32_t reserved[7];
};
static_assert(sizeof(util_agg_hdr_view_t) == sizeof(util_agg_msg_t), "host view must match the device header");

// Offset of journal[] from the base of the landing region. The aggregator
// computes a destination as UTIL_AGG_JOURNAL_OFFSET + (head % capacity) * 32,
// which is 32 B-aligned for every head -- see the alignment note on
// util_agg_entry_t.
constexpr uint32_t UTIL_AGG_JOURNAL_OFFSET = sizeof(util_agg_msg_t);
static_assert(UTIL_AGG_JOURNAL_OFFSET % 16 == 0);
