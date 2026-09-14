// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qsr.s1 ATT NoC probe (device side of Qsr1AttNocProbe in test_single_dm_l1_write.cpp).
//
// One DM kernel on logical {0,0} walks a fixed list of NoC operations, one operation class per step,
// and after EACH step writes a step-done marker (0xD0DE0000 | step) into an uncached per-step slot of
// its own L1 result block. The host polls those slots, so a hang localizes to the first step whose
// marker never appears. Before issuing any NoC traffic the kernel records every 64-bit NoC operand it
// is about to use (unicast, DRAM via both address-generation paths, semaphore, both multicast
// descriptors), so the host can decode which ATT window/selector each resolved to even if a later
// step hangs. No DPRINT (unsafe on this model): everything goes through the result block.
//
// Step table (a step runs only if bit `step` of step_mask is set; a skipped step writes 0x5C1B0000|step):
//   0  header: self-identity + every resolved NoC operand (pure computation, no NoC traffic)
//   1  local write of a 16 B sentinel into this tile's own L1
//   2  unicast write 16 B -> worker (dst_x, dst_y) @ result_addr+0x300, write barrier
//   3  unicast read 16 B <- worker (dst_x, dst_y) @ result_addr+0x340 (host pre-seeded), read barrier
//   4  DRAM write 16 B via InterleavedAddrGen<true> (ATT Address::dram path) @ dram_addr, write barrier
//   5  DRAM read 16 B back via the same addrgen, read barrier
//   6  noc_semaphore_inc(+1) on worker (dst_x, dst_y)'s probe semaphore, atomic barrier
//   7  multicast write 16 B to the column rectangle (col_*), num_dests = col_num_dests, write barrier
//   8  multicast write 16 B to the grid rectangle (grid_*), num_dests = grid_num_dests, write barrier
//   9  DRAM write 16 B via get_noc_addr_from_bank_id<true> (host bank table -> resolve_current path)
//      @ dram_addr_alt, write barrier   [EXTRA: the two DRAM address paths resolve differently under ATT]
//  10  checkpoint marker: steps 0..9 walked (the original probe's end; always written, not mask-gated)
//  -- read-size bracket: ONE noc_async_read of the given size, read barrier, record first/last word --
//  11  DRAM read  4096 B via InterleavedAddrGen<true> @ dram_rd_addr (host-seeded 32 KiB pattern)
//  12  DRAM read  8128 B  (largest 64 B multiple below one 8192 B overlay packet)
//  13  DRAM read  8192 B  (== NOC_OVERLAY_MAX_BYTES_IN_PACKET: exactly one full packet)
//  14  DRAM read  8256 B  (one full packet + 64 B: two packets)
//  15  DRAM read 16384 B  (two full packets)
//  16  L1 read    8192 B <- worker (dst_x, dst_y) @ result_addr+0x30000 (host-seeded per tile): same size as
//      s13 but a worker endpoint, separating 'DRAM endpoint' from 'read path'
//  17  final done marker
//  Sizes ascend so a hang at s13 with s12 done pins '== 8192'; a hang at s12 pins '> 4096'. The V3 issue path
//  hands the whole length to the overlay in one command (no software chunking); the overlay packetizes at
//  8192 B, so s14/s15 are two-packet reads.
//
// Result block layout (byte offsets from result_addr; every slot 64 B aligned; mirrored on the host):
//   0x000 header (HDR_* below)
//   0x080 step markers: word at 0x080 + 4*step, step 0..17
//   0x100 s2 payload (unicast source)         0x140 s4 payload (DRAM addrgen source)
//   0x180 s7 payload (mcast column source)    0x1C0 s8 payload (mcast grid source)
//   0x200 s9 payload (DRAM bank-id source)    0x240 s3 read landing (local)
//   0x280 s5 read landing (local)             0x2C0 s1 local sentinel slot
//   -- remote landing slots, same offsets on EVERY worker (host seeds sentinels) --
//   0x300 s2 unicast landing                  0x340 s3 remote source (host pre-seeded per tile)
//   0x380 s7 mcast column landing             0x3C0 s8 mcast grid landing
//   -- read-size bracket (src worker only unless noted) --
//   0x400 + 0x40*k  record for step 11+k (REC_* below: size, first, last, src operand, dst, phase)
//   0x10000 + 0x4000*k  16 KiB landing region for step 11+k (host zeroes; the host counts delivered words)
//   0x30000  8 KiB host-seeded source pattern on EVERY worker (s16 reads the dst worker's copy)

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace qsr1_att_noc_probe {

constexpr uint32_t OFF_HDR = 0x000;
constexpr uint32_t OFF_STEP = 0x080;
constexpr uint32_t OFF_SRC_S2 = 0x100;
constexpr uint32_t OFF_SRC_S4 = 0x140;
constexpr uint32_t OFF_SRC_S7 = 0x180;
constexpr uint32_t OFF_SRC_S8 = 0x1C0;
constexpr uint32_t OFF_SRC_S9 = 0x200;
constexpr uint32_t OFF_DST_S3 = 0x240;
constexpr uint32_t OFF_DST_S5 = 0x280;
constexpr uint32_t OFF_LOCAL_S1 = 0x2C0;
constexpr uint32_t OFF_REMOTE_S2 = 0x300;
constexpr uint32_t OFF_REMOTE_S3_SRC = 0x340;
constexpr uint32_t OFF_REMOTE_S7 = 0x380;
constexpr uint32_t OFF_REMOTE_S8 = 0x3C0;

// Header word offsets (bytes from result_addr + OFF_HDR). 64-bit values are (lo, hi) pairs.
constexpr uint32_t HDR_MAGIC = 0x00;
constexpr uint32_t HDR_SELF = 0x04;            // my_x | my_y << 8 | noc_index << 16
constexpr uint32_t HDR_NODE_ID = 0x08;         // raw NOC_NODE_ID register
constexpr uint32_t HDR_SEM_ADDR = 0x0C;        // raw L1 offset of the probe semaphore
constexpr uint32_t HDR_UNICAST_ADDR = 0x10;    // s2 destination operand
constexpr uint32_t HDR_DRAM_GEN_ADDR = 0x18;   // s4/s5 InterleavedAddrGen<true> operand
constexpr uint32_t HDR_DRAM_BANK_ADDR = 0x20;  // s9 get_noc_addr_from_bank_id<true> operand
constexpr uint32_t HDR_SEM_NOC_ADDR = 0x28;    // s6 semaphore operand
constexpr uint32_t HDR_MCAST_COL = 0x30;       // s7 multicast descriptor
constexpr uint32_t HDR_MCAST_GRID = 0x38;      // s8 multicast descriptor
constexpr uint32_t HDR_BANK_XY = 0x40;         // dram_bank_to_noc_xy[noc_index][bank]
constexpr uint32_t HDR_BANK_OFF = 0x44;        // bank_to_dram_offset[bank]
constexpr uint32_t HDR_S3_VALUE = 0x48;        // first word read from the remote worker in s3
constexpr uint32_t HDR_S5_VALUE = 0x4C;        // first word read back from DRAM in s5
constexpr uint32_t HDR_STEP_MASK = 0x50;
constexpr uint32_t HDR_NUM_DRAM_BANKS = 0x54;
constexpr uint32_t HDR_S3_SRC_ADDR = 0x58;  // s3 source operand

constexpr uint32_t MAGIC = 0xA77B10C0u;
constexpr uint32_t STEP_DONE_TAG = 0xD0DE0000u;
constexpr uint32_t STEP_SKIP_TAG = 0x5C1B0000u;
constexpr uint32_t XFER_BYTES = 16;
constexpr uint32_t ORIGINAL_END_STEP = 10;  // unconditional marker: steps 0..9 walked
constexpr uint32_t FINAL_STEP = 17;

// ---- read-size bracket (steps 11..16) ----
constexpr uint32_t RD_FIRST_STEP = 11;
constexpr uint32_t RD_STEPS = 6;
constexpr uint32_t RD_L1_STEP = 16;  // source is the other worker's L1, not DRAM
constexpr uint32_t RD_SIZES[RD_STEPS] = {4096, 8128, 8192, 8256, 16384, 8192};
constexpr uint32_t OFF_RD_REC = 0x400;  // per-step 64 B records, [0x400, 0x580)
constexpr uint32_t RD_REC_STRIDE = 0x40;
constexpr uint32_t OFF_RD_LANDING = 0x10000;  // per-step 16 KiB landing regions, [0x10000, 0x28000)
constexpr uint32_t RD_LANDING_STRIDE = 0x4000;
constexpr uint32_t OFF_RD_L1_SRC = 0x30000;  // host-seeded 8 KiB on every worker: the s16 source
// Record word offsets (bytes from the record base).
constexpr uint32_t REC_SIZE = 0x00;
constexpr uint32_t REC_FIRST = 0x04;  // first landing word after the barrier
constexpr uint32_t REC_LAST = 0x08;   // last landing word after the barrier
constexpr uint32_t REC_SRC = 0x0C;    // 64-bit source operand (lo, hi)
constexpr uint32_t REC_DST = 0x14;    // local landing address
constexpr uint32_t REC_PHASE = 0x18;  // PHASE_*: where the step was when the host looked
constexpr uint32_t PHASE_PENDING = 0;
constexpr uint32_t PHASE_SKIPPED = 1;
constexpr uint32_t PHASE_ISSUING = 2;  // about to call noc_async_read
constexpr uint32_t PHASE_ISSUED = 3;   // noc_async_read returned; inside noc_async_read_barrier
constexpr uint32_t PHASE_DONE = 4;     // barrier returned

// Uncached L1 alias: bypasses the L1 D$/L2 so the host (and the NoC) see every store immediately.
inline volatile tt_l1_ptr uint32_t* uncached(uint32_t l1_byte_addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_byte_addr + MEM_L1_UNCACHED_BASE);
}
inline void put32(uint32_t l1_byte_addr, uint32_t v) { *uncached(l1_byte_addr) = v; }
inline uint32_t get32(uint32_t l1_byte_addr) { return *uncached(l1_byte_addr); }
inline void put64(uint32_t l1_byte_addr, uint64_t v) {
    put32(l1_byte_addr, static_cast<uint32_t>(v));
    put32(l1_byte_addr + 4, static_cast<uint32_t>(v >> 32));
}

inline uint32_t payload_word(uint32_t step, uint32_t i) { return 0x51DE0000u | (step << 8) | i; }

inline void fill_payload(uint32_t l1_byte_addr, uint32_t step) {
    for (uint32_t i = 0; i < XFER_BYTES / sizeof(uint32_t); ++i) {
        put32(l1_byte_addr + 4 * i, payload_word(step, i));
    }
    __asm__ __volatile__("fence" ::: "memory");
}

inline void mark_done(uint32_t base, uint32_t step) {
    __asm__ __volatile__("fence" ::: "memory");
    put32(base + OFF_STEP + 4 * step, STEP_DONE_TAG | step);
}
inline void mark_skip(uint32_t base, uint32_t step) { put32(base + OFF_STEP + 4 * step, STEP_SKIP_TAG | step); }
inline void set_phase(uint32_t rec, uint32_t phase) {
    __asm__ __volatile__("fence" ::: "memory");
    put32(rec + REC_PHASE, phase);
}
inline bool enabled(uint32_t mask, uint32_t step) { return ((mask >> step) & 1u) != 0; }

}  // namespace qsr1_att_noc_probe

void kernel_main() {
    using namespace qsr1_att_noc_probe;

    const uint32_t base = get_arg(args::result_addr);
    const uint32_t dst_x = get_arg(args::dst_x);
    const uint32_t dst_y = get_arg(args::dst_y);
    const uint32_t bank = get_arg(args::dram_bank_id);
    const uint32_t dram_addr = get_arg(args::dram_addr);
    const uint32_t dram_addr_alt = get_arg(args::dram_addr_alt);
    const uint32_t dram_rd_addr = get_arg(args::dram_rd_addr);
    const uint32_t col_x0 = get_arg(args::col_x0);
    const uint32_t col_y0 = get_arg(args::col_y0);
    const uint32_t col_x1 = get_arg(args::col_x1);
    const uint32_t col_y1 = get_arg(args::col_y1);
    const uint32_t col_num_dests = get_arg(args::col_num_dests);
    const uint32_t grid_x0 = get_arg(args::grid_x0);
    const uint32_t grid_y0 = get_arg(args::grid_y0);
    const uint32_t grid_x1 = get_arg(args::grid_x1);
    const uint32_t grid_y1 = get_arg(args::grid_y1);
    const uint32_t grid_num_dests = get_arg(args::grid_num_dests);
    const uint32_t step_mask = get_arg(args::step_mask);

    // ---- step 0: header. Every operand is computed here, before any NoC traffic, so the host can
    // decode window/selector choices even if a later step never completes.
    const uint32_t sem_addr = static_cast<uint32_t>(get_semaphore(sem::probe_sem));
    const uint64_t unicast_dst = get_noc_addr(dst_x, dst_y, base + OFF_REMOTE_S2, noc_index);
    const uint64_t s3_src = get_noc_addr(dst_x, dst_y, base + OFF_REMOTE_S3_SRC, noc_index);
    const InterleavedAddrGen<true> dram_gen{.bank_base_address = dram_addr, .page_size = 64};
    const uint64_t dram_gen_addr = dram_gen.get_noc_addr(0, 0, noc_index);
    const uint64_t dram_bank_addr = get_noc_addr_from_bank_id<true>(bank, dram_addr_alt, noc_index);
    const uint64_t sem_noc = get_noc_addr(dst_x, dst_y, sem_addr, noc_index);
    const uint64_t mcast_col = get_noc_multicast_addr(col_x0, col_y0, col_x1, col_y1, base + OFF_REMOTE_S7, noc_index);
    const uint64_t mcast_grid =
        get_noc_multicast_addr(grid_x0, grid_y0, grid_x1, grid_y1, base + OFF_REMOTE_S8, noc_index);

    const uint32_t hdr = base + OFF_HDR;
    put32(hdr + HDR_SELF,
          static_cast<uint32_t>(my_x[noc_index]) | (static_cast<uint32_t>(my_y[noc_index]) << 8) |
              (static_cast<uint32_t>(noc_index) << 16));
    put32(hdr + HDR_NODE_ID, NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID));
    put32(hdr + HDR_SEM_ADDR, sem_addr);
    put64(hdr + HDR_UNICAST_ADDR, unicast_dst);
    put64(hdr + HDR_DRAM_GEN_ADDR, dram_gen_addr);
    put64(hdr + HDR_DRAM_BANK_ADDR, dram_bank_addr);
    put64(hdr + HDR_SEM_NOC_ADDR, sem_noc);
    put64(hdr + HDR_MCAST_COL, mcast_col);
    put64(hdr + HDR_MCAST_GRID, mcast_grid);
    put32(hdr + HDR_BANK_XY, static_cast<uint32_t>(dram_bank_to_noc_xy[noc_index][bank]));
    put32(hdr + HDR_BANK_OFF, static_cast<uint32_t>(bank_to_dram_offset[bank]));
    put32(hdr + HDR_S3_VALUE, 0);
    put32(hdr + HDR_S5_VALUE, 0);
    put32(hdr + HDR_STEP_MASK, step_mask);
    put32(hdr + HDR_NUM_DRAM_BANKS, NUM_DRAM_BANKS);
    put64(hdr + HDR_S3_SRC_ADDR, s3_src);
    // Read-size bracket records: size, source operand and landing address of every bracket step, written here
    // (before any NoC traffic) so the host can decode them even if an earlier step hangs.
    const InterleavedAddrGen<true> rd_gen{.bank_base_address = dram_rd_addr, .page_size = 64};
    const uint64_t rd_dram_src = rd_gen.get_noc_addr(0, 0, noc_index);
    const uint64_t rd_l1_src = get_noc_addr(dst_x, dst_y, base + OFF_RD_L1_SRC, noc_index);
    for (uint32_t k = 0; k < RD_STEPS; ++k) {
        const uint32_t rec = base + OFF_RD_REC + RD_REC_STRIDE * k;
        put32(rec + REC_SIZE, RD_SIZES[k]);
        put32(rec + REC_FIRST, 0);
        put32(rec + REC_LAST, 0);
        put64(rec + REC_SRC, (RD_FIRST_STEP + k == RD_L1_STEP) ? rd_l1_src : rd_dram_src);
        put32(rec + REC_DST, base + OFF_RD_LANDING + RD_LANDING_STRIDE * k);
        put32(rec + REC_PHASE, PHASE_PENDING);
    }
    put32(hdr + HDR_MAGIC, MAGIC);
    mark_done(base, 0);

    // ---- step 1: local write of a sentinel (no NoC).
    if (enabled(step_mask, 1)) {
        fill_payload(base + OFF_LOCAL_S1, 1);
        mark_done(base, 1);
    } else {
        mark_skip(base, 1);
    }

    // ---- step 2: unicast write to the other live worker + write barrier.
    if (enabled(step_mask, 2)) {
        fill_payload(base + OFF_SRC_S2, 2);
        noc_async_write(base + OFF_SRC_S2, unicast_dst, XFER_BYTES, noc_index);
        noc_async_write_barrier(noc_index);
        mark_done(base, 2);
    } else {
        mark_skip(base, 2);
    }

    // ---- step 3: unicast read from the other live worker (host pre-seeded) + read barrier.
    if (enabled(step_mask, 3)) {
        noc_async_read(s3_src, base + OFF_DST_S3, XFER_BYTES, noc_index);
        noc_async_read_barrier(noc_index);
        put32(hdr + HDR_S3_VALUE, get32(base + OFF_DST_S3));
        mark_done(base, 3);
    } else {
        mark_skip(base, 3);
    }

    // ---- step 4: DRAM write via the interleaved address generator (ATT Address::dram path).
    if (enabled(step_mask, 4)) {
        fill_payload(base + OFF_SRC_S4, 4);
        noc_async_write(base + OFF_SRC_S4, dram_gen_addr, XFER_BYTES, noc_index);
        noc_async_write_barrier(noc_index);
        mark_done(base, 4);
    } else {
        mark_skip(base, 4);
    }

    // ---- step 5: DRAM read back via the same address generator.
    if (enabled(step_mask, 5)) {
        noc_async_read(dram_gen_addr, base + OFF_DST_S5, XFER_BYTES, noc_index);
        noc_async_read_barrier(noc_index);
        put32(hdr + HDR_S5_VALUE, get32(base + OFF_DST_S5));
        mark_done(base, 5);
    } else {
        mark_skip(base, 5);
    }

    // ---- step 6: remote atomic increment of the other worker's probe semaphore + atomic barrier.
    if (enabled(step_mask, 6)) {
        noc_semaphore_inc(sem_noc, 1, noc_index);
        noc_async_atomic_barrier(noc_index);
        mark_done(base, 6);
    } else {
        mark_skip(base, 6);
    }

    // ---- step 7: multicast to the column rectangle (self excluded: plain variant, no SRC_INCLUDE).
    if (enabled(step_mask, 7)) {
        fill_payload(base + OFF_SRC_S7, 7);
        noc_async_write_multicast(base + OFF_SRC_S7, mcast_col, XFER_BYTES, col_num_dests, false, noc_index);
        noc_async_write_barrier(noc_index);
        mark_done(base, 7);
    } else {
        mark_skip(base, 7);
    }

    // ---- step 8: multicast to the full worker-grid rectangle (self excluded).
    if (enabled(step_mask, 8)) {
        fill_payload(base + OFF_SRC_S8, 8);
        noc_async_write_multicast(base + OFF_SRC_S8, mcast_grid, XFER_BYTES, grid_num_dests, false, noc_index);
        noc_async_write_barrier(noc_index);
        mark_done(base, 8);
    } else {
        mark_skip(base, 8);
    }

    // ---- step 9 (extra): DRAM write via the bank-id path (host bank table -> resolve_current).
    if (enabled(step_mask, 9)) {
        fill_payload(base + OFF_SRC_S9, 9);
        noc_async_write(base + OFF_SRC_S9, dram_bank_addr, XFER_BYTES, noc_index);
        noc_async_write_barrier(noc_index);
        mark_done(base, 9);
    } else {
        mark_skip(base, 9);
    }

    // ---- step 10: unconditional checkpoint marker, the original probe's end (steps 0..9 walked).
    mark_done(base, ORIGINAL_END_STEP);

    // ---- steps 11..16: read-size bracket. Ascending DRAM sizes (4096, 8128, 8192, 8256, 16384), then 8192 B
    // from the other worker's L1. One noc_async_read per step into its own landing region, then a read barrier,
    // then the first/last landing word is recorded. The phase word tells the host whether a hung step is stuck
    // issuing the read or waiting in the barrier.
    for (uint32_t k = 0; k < RD_STEPS; ++k) {
        const uint32_t step = RD_FIRST_STEP + k;
        const uint32_t rec = base + OFF_RD_REC + RD_REC_STRIDE * k;
        if (!enabled(step_mask, step)) {
            set_phase(rec, PHASE_SKIPPED);
            mark_skip(base, step);
            continue;
        }
        const uint32_t size = RD_SIZES[k];
        const uint32_t dst = base + OFF_RD_LANDING + RD_LANDING_STRIDE * k;
        const uint64_t src = (step == RD_L1_STEP) ? rd_l1_src : rd_dram_src;
        set_phase(rec, PHASE_ISSUING);
        noc_async_read(src, dst, size, noc_index);
        set_phase(rec, PHASE_ISSUED);
        noc_async_read_barrier(noc_index);
        set_phase(rec, PHASE_DONE);
        put32(rec + REC_FIRST, get32(dst));
        put32(rec + REC_LAST, get32(dst + size - sizeof(uint32_t)));
        mark_done(base, step);
    }

    // ---- step 17: final marker.
    mark_done(base, FINAL_STEP);
}
