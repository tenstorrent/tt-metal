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

#include <cstddef>
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

// Header is 64 B so journal[] starts 16 B-aligned.
//
// FIELD ORDER IS LOAD-BEARING. A host reading this header over the NON_MMIO tunnel does
// NOT get an atomic 64 B snapshot -- the read is served in 16 B chunks that can come
// from different moments. Measured on a T3K: `head` (offset 8) and a checksum at offset
// 32 arrived from different publishes, every time, with every field individually sane.
//
// So the only field that changes fast -- `head` -- lives in the FIRST 16 B chunk next
// to its own tear detector, and nothing else volatile shares that chunk. A reader that
// sees `head ^ UTIL_AGG_HEAD_SALT == head_xor` has a good head, whatever happened to
// the rest of the header. Everything after chunk 0 is either static after init or
// advisory.
//
// Do not "tidy" this by grouping fields logically. The grouping IS the correctness
// argument.
constexpr uint32_t UTIL_AGG_HEAD_SALT = 0xA5A5A5A5u;

struct util_agg_msg_t {
    // --- chunk 0: the fast-changing field and its self-check. Nothing else. ---
    volatile uint32_t magic;     // 'TTAG'
    volatile uint32_t version;   // 1
    volatile uint32_t head;      // monotonic COUNT of entries ever written
    volatile uint32_t head_xor;  // head ^ UTIL_AGG_HEAD_SALT, written immediately after head

    // --- chunk 1: static after init, guarded by hdr_checksum ---
    volatile uint32_t capacity;      // entries in journal[]; head % capacity is the write slot
    volatile uint32_t num_cores;     // cores this aggregator sweeps. NEVER assume 64 (see 2.1)
    volatile uint32_t src_chip;      // PHYSICAL chip id of the chip this journal describes
    volatile uint32_t hdr_checksum;  // over the static fields only -- a layout sanity check

    // --- chunk 2: advisory. A torn value here costs nothing. ---
    volatile uint32_t sweep_count;  // liveness heartbeat; the host's staleness check keys on this
    volatile uint32_t lost;         // entries dropped because a Tensix ring wrapped before we swept it
    volatile uint32_t reserved[6];

    volatile util_agg_entry_t journal[];
};
static_assert(sizeof(util_agg_msg_t) == 64);
static_assert(sizeof(util_agg_msg_t) % 16 == 0, "journal[] must start 16 B-aligned");
static_assert(offsetof(util_agg_msg_t, head) < 16, "head must live in the first 16 B chunk");
static_assert(offsetof(util_agg_msg_t, head_xor) < 16, "head_xor must share head's chunk");

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

// Checksum over the STATIC fields only. Those never change after init, so this is a
// layout/sanity check across chunks, not a tear detector -- tearing is handled by
// head_xor inside chunk 0.
inline uint32_t util_agg_hdr_checksum(
    uint32_t magic, uint32_t version, uint32_t capacity, uint32_t num_cores, uint32_t src_chip) {
    return magic + version + capacity + num_cores + src_chip;
}

// PUBLISH ORDERING CONTRACT
//
// Per sweep the aggregator: (1) barriers its NOC reads so every sample is committed to
// the journal, (2) writes `head`, then `head_xor` immediately after, (3) updates the
// advisory fields. The static fields and hdr_checksum are written once at init.
//
// A host reads the header and accepts `head` iff magic is right and
// head ^ UTIL_AGG_HEAD_SALT == head_xor. On mismatch it re-reads; it must NOT fall back
// to a previous head, because those entries are committed and skipping them loses
// samples.
//
// Entries below `head` are committed, but the ring still wraps: a host that dawdles
// between reading the header and reading the entries can have the oldest slots
// overwritten underneath it. At the default sizing that is ~96 ms of slack; read
// promptly, and re-read `head` afterwards if you need to prove you were not lapped.

// Plain host-side view of the header.
//
// util_agg_msg_t's fields are volatile -- they are written by the aggregator and polled
// by a host that must not have them cached -- which makes the struct non-trivially
// copyable and therefore not a legal memcpy destination. This is the identical layout
// without the qualifier, for host code that reads a journal into a buffer.
struct util_agg_hdr_view_t {
    uint32_t magic;
    uint32_t version;
    uint32_t head;
    uint32_t head_xor;
    uint32_t capacity;
    uint32_t num_cores;
    uint32_t src_chip;
    uint32_t hdr_checksum;
    uint32_t sweep_count;
    uint32_t lost;
    uint32_t reserved[6];
};

// Did this header read cleanly? Chunk 0 is self-validating, which is the whole point.
inline bool util_agg_hdr_ok(const util_agg_hdr_view_t& h) {
    return h.magic == UTIL_AGG_MAGIC && (h.head ^ UTIL_AGG_HEAD_SALT) == h.head_xor;
}
static_assert(sizeof(util_agg_hdr_view_t) == sizeof(util_agg_msg_t), "host view must match the device header");

// Offset of journal[] from the base of the journal region. The aggregator writes entry
// `head % capacity` at UTIL_AGG_JOURNAL_OFFSET + slot * 32, which is 32 B-aligned for
// every head -- see the alignment note on util_agg_entry_t.
constexpr uint32_t UTIL_AGG_JOURNAL_OFFSET = sizeof(util_agg_msg_t);
static_assert(UTIL_AGG_JOURNAL_OFFSET % 16 == 0);
