// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * socket_layout.h - LIM layout for X280 H2D / D2H sockets and the
 *                   migration worker.
 *
 * Defines the byte offsets within L3 LIM where the migration control
 * mailboxes, DRAM bank table, socket slot table + per-socket configs,
 * staging buffer, and H2D-FIFO bump pool live. Shared by X280 firmware
 * and host code that allocates sockets and writes their configs.
 *
 * Architecture
 * ------------
 *
 * The host sizes per-socket FIFOs and constructs sockets; the firmware
 * reads each socket's config address from a fixed-layout slot table in
 * LIM. The host fills the slot table before flipping CONFIGS_READY.
 * The firmware knows two things at compile time:
 *   1. The slot table address (X280_SOCKET_SLOT_TABLE_ADDR).
 *   2. Which slot index corresponds to which logical role
 *      (X280_SOCKET_ROLE_*).
 *
 * Slot table contract
 * -------------------
 *
 * Five slots, hard-capped (X280_MAX_SOCKETS = 5). Each slot is two
 * 32-bit fields, totalling 8 bytes:
 *
 *   slot[i] = { uint32_t config_addr;   // LIM addr of socket cfg blob
 *               uint32_t fifo_addr; }   // LIM addr of H2D HOST_PUSH ring
 *
 * The roles are:
 *   slot 0  X280_SOCKET_ROLE_H2D_CMD          (HOST_PUSH)
 *   slot 1  X280_SOCKET_ROLE_D2H              (D2H)
 *   slot 2  X280_SOCKET_ROLE_H2D_DATA         (DEVICE_PULL or HOST_PUSH)
 *   slot 3  X280_SOCKET_ROLE_D2H_COMPLETION   (D2H, reserved)
 *   slot 4  X280_SOCKET_ROLE_SPARE            (reserved)
 *
 * The host MAY leave any slot zeroed if its firmware does not use
 * that role. The firmware MUST NOT touch a zeroed slot.
 *
 * config_addr for an H2D HOST_PUSH socket points at a
 * receiver_socket_md blob the host pre-writes; fifo_addr points
 * into the H2D-FIFO bump pool (where the firmware will read the
 * payload pages). For an H2D DEVICE_PULL socket fifo_addr is a
 * logical anchor only -- the actual data lives in pinned host RAM
 * named by the cfg blob's data_addr_hi/lo fields. For a D2H socket
 * fifo_addr is unused (zeroed); the FIFO lives in pinned host RAM.
 *
 * LIM layout (LIM is 1.875 MiB at X280_LIM_BASE = 0x08000000)
 * ------------------------------------------------------
 *
 *   0x000000 firmware .text/.data/.bss       (~tens of KiB)
 *   ...
 *   0x100000 X280_SENTINEL_ADDR              (x280_address_map.h)
 *   0x120000 spike mailbox region            (PCIe NoC-write spike)
 *
 *   0x130000 migration control mailboxes     (8 KiB; STATUS / CMD_COUNT /
 *                                              CONFIGS_READY / DATA_PAGE_SIZE
 *                                              on disjoint cache lines)
 *   0x132000 DRAM bank table                 (4 KiB; host pre-writes
 *                                              num_banks + (noc_xy_enc,
 *                                              base_addr, bank_size, _) per
 *                                              bank)
 *   0x133000 socket slot table               (5 entries x 8 B = 40 B)
 *   0x133040 socket configs                  (5 x 128 B = 640 B,
 *                                              cache-line aligned;
 *                                              host pre-writes a
 *                                              receiver_socket_md /
 *                                              sender_socket_md+ack+enc
 *                                              blob into each used slot)
 *   0x134000 staging buffer                  (X280_STAGING_SIZE,
 *                                              default 64 KiB; LIM scratch
 *                                              for chunked DRAM <-> data-
 *                                              ring copies)
 *   0x144000 H2D-FIFO bump pool              (X280_H2D_FIFO_POOL_SIZE,
 *                                              default 64 KiB; host
 *                                              allocator hands out per-
 *                                              H2D-HOST_PUSH ring slices)
 *   ...      (free for future expansion)
 *   0x1D8000 stack bottom (top - 32 KiB)     (boot/entry.S, x280.ld)
 *   0x1E0000 LIM end (X280_LIM_END)
 *
 * Total reservation: ~144 KiB. The placement order puts small fixed-
 * size control structures (mailboxes, bank table, slot table +
 * configs) at predictable low offsets, with the two large buffers
 * (staging, FIFO pool) at the high end so growing either is a one-
 * line offset bump that doesn't disturb anything else.
 *
 * Cache-line discipline
 * ---------------------
 *
 * Every control mailbox in the migration-control block lives on its
 * own 64-byte cache line so:
 *   - Host partial-word writes don't false-share with firmware loads.
 *   - The cache controller's read-merge-write cycle on first write
 *     doesn't perturb an unrelated control field.
 * Host setup primes each cache line with a full 64 B zero write before
 * any partial-word write happens, satisfying L3 ECC.
 *
 * All offsets here form the wire contract between host setup code
 * and X280 firmware. Every host and firmware consumer must update in
 * lock-step when an offset changes.
 */
#ifndef SOCKET_LAYOUT_H
#define SOCKET_LAYOUT_H

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <stdint.h>

/* These headers are shared by C firmware and C++17 translation units.
 * Map the C++ spelling to C11 without pulling <assert.h> into
 * freestanding firmware. */
#ifndef __cplusplus
#if !defined(static_assert)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#endif
#endif

/* The spike firmware owns a disjoint mailbox reservation. */
#define X280_SPIKE_MBOX_OFFSET 0x00120000UL
#define X280_SPIKE_MBOX_SIZE 0x00010000UL /* 64 KiB reserved */

/* ------------------------------------------------------------------
 * Migration control mailboxes (region 1, lowest offset).
 *
 * Four mailboxes, each on its own 64-byte cache line so partial-word
 * writes on either side don't false-share. Layout inside this region:
 *   +0x00  STATUS         (firmware -> host; INIT / READY / RUNNING /
 *                          STOPPED / ERROR_*)
 *   +0x80  CMD_COUNT      (firmware -> host; processed cmd / page count)
 *   +0xC0  CONFIGS_READY  (host -> firmware; MIGRATION_CONFIGS_READY_VALUE)
 *   +0x100 DATA_PAGE_SIZE (host -> firmware; H2D-data + D2H page size)
 *
 * Optional benchmark counters published only when the migration worker
 * is built with -DX280_BENCHMARK. Two directions
 * (READ / WRITE) x 5 mailboxes each (cycles_lo, cycles_hi, bytes_lo,
 * bytes_hi, cmd_count); each on its own cache line. Offsets exist
 * unconditionally so host-side readers can reference them, but the
 * firmware leaves them at zero when X280_BENCHMARK is not defined.
 *
 * UNITS NOTE: every *_CYCLES_* mailbox below holds CLINT mtime ticks
 * at 50 MHz, not rdcycle ticks at PLL frequency. rdcycle does not advance
 * during CPU stalls (MEASUREMENT.md
 * "Unconfirmed hardware assumptions", item 5). Host readers
 * divide by 50_000_000 to get seconds. The *_CYCLES_* names are fixed
 * wire identifiers despite carrying mtime ticks.
 *
 * Offsets +0x180..+0x2FF are unallocated. Bench mailboxes start at
 * +0x300.
 *
 *   +0x300 BENCH_READ_CYCLES_LO   (firmware -> host; u64 lo half)
 *   +0x340 BENCH_READ_CYCLES_HI   (firmware -> host; u64 hi half)
 *   +0x380 BENCH_READ_BYTES_LO    (firmware -> host; u64 lo half)
 *   +0x3C0 BENCH_READ_BYTES_HI    (firmware -> host; u64 hi half)
 *   +0x400 BENCH_READ_CMD_COUNT   (firmware -> host; u32)
 *   +0x440 BENCH_WRITE_CYCLES_LO  (firmware -> host; u64 lo half)
 *   +0x480 BENCH_WRITE_CYCLES_HI  (firmware -> host; u64 hi half)
 *   +0x4C0 BENCH_WRITE_BYTES_LO   (firmware -> host; u64 lo half)
 *   +0x500 BENCH_WRITE_BYTES_HI   (firmware -> host; u64 hi half)
 *   +0x540 BENCH_WRITE_CMD_COUNT  (firmware -> host; u32)
 *
 * Split-timer counters (per-hop) for attributing the READ/WRITE
 * asymmetry. Each cmd has two NOC hops; we time each hop separately
 * and accumulate. Same lo/hi u64 split convention as the totals above.
 *   +0x580 BENCH_READ_DRAM_CYCLES_LO   (READ cmd: dram_read_block hop)
 *   +0x5C0 BENCH_READ_DRAM_CYCLES_HI
 *   +0x600 BENCH_READ_PCIE_CYCLES_LO   (READ cmd: socket_write_payload hop)
 *   +0x640 BENCH_READ_PCIE_CYCLES_HI
 *   +0x680 BENCH_WRITE_PCIE_CYCLES_LO  (WRITE cmd: socket_pull_payload hop)
 *   +0x6C0 BENCH_WRITE_PCIE_CYCLES_HI
 *   +0x700 BENCH_WRITE_DRAM_CYCLES_LO  (WRITE cmd: dram_write_block hop)
 *   +0x740 BENCH_WRITE_DRAM_CYCLES_HI
 *
 * Socket-page wait counters. Bracket the per-page producer/consumer
 * sync point so we can tell when the firmware is starved waiting for
 * the host vs actually moving bytes.
 *   +0x780 BENCH_READ_SOCKET_WAIT_CYCLES_LO   (READ cmd:
 *                                              socket_reserve_pages
 *                                              waiting for D2H FIFO
 *                                              reservation slot)
 *   +0x7C0 BENCH_READ_SOCKET_WAIT_CYCLES_HI
 *   +0x800 BENCH_WRITE_SOCKET_WAIT_CYCLES_LO  (WRITE cmd:
 *                                              socket_wait_for_pages
 *                                              waiting for host to
 *                                              push an H2D-data page)
 *   +0x840 BENCH_WRITE_SOCKET_WAIT_CYCLES_HI
 *
 * Multi-hart dispatch mailboxes are used only by
 * X280_XFER_MODE=SCALAR builds. Hart 0 parks
 * helper harts in multihart_helper_loop polling these. See the
 * dedicated block below for the full layout.
 *   +0x900 MULTIHART_WORK_PRESENT
 *   +0x940 MULTIHART_HART_DONE[0..3]   (4 cache lines, +0x940..+0xA00)
 *   +0xA40 MULTIHART_WORK_DESC[0..3]   (4 cache lines, +0xA40..+0xB00)
 *
 * The 8 KiB reservation includes the C++ runtime's per-hart lifecycle
 * records at +0x1800..+0x1AFF. Offsets +0x180..+0x2FF and
 * +0xD00..+0xDFF are unallocated.
 * ------------------------------------------------------------------ */
#define X280_MIGRATION_CONTROL_OFFSET 0x00130000UL
#define X280_MIGRATION_CONTROL_SIZE 0x00002000UL /* 8 KiB reserved */
#define X280_MIGRATION_CONTROL_ADDR (X280_LIM_BASE + X280_MIGRATION_CONTROL_OFFSET)

/* Primary migration mailboxes. The +0x80 progress slot reports completed
 * commands in the migration worker and echoed pages in the socket echo. */
#define X280_MIGRATION_STATUS_OFFSET 0x00UL
#define X280_MIGRATION_CMD_COUNT_OFFSET 0x80UL
#define X280_MIGRATION_CONFIGS_READY_OFFSET 0xC0UL
#define X280_MIGRATION_DATA_PAGE_SIZE_OFFSET 0x100UL

#define X280_MIGRATION_STATUS_ADDR ((volatile uint32_t*)(X280_MIGRATION_CONTROL_ADDR + X280_MIGRATION_STATUS_OFFSET))
#define X280_MIGRATION_CMD_COUNT_ADDR \
    ((volatile uint32_t*)(X280_MIGRATION_CONTROL_ADDR + X280_MIGRATION_CMD_COUNT_OFFSET))
#define X280_MIGRATION_CONFIGS_READY_ADDR \
    ((volatile uint32_t*)(X280_MIGRATION_CONTROL_ADDR + X280_MIGRATION_CONFIGS_READY_OFFSET))
#define X280_MIGRATION_DATA_PAGE_SIZE_ADDR \
    ((volatile uint32_t*)(X280_MIGRATION_CONTROL_ADDR + X280_MIGRATION_DATA_PAGE_SIZE_OFFSET))

#define X280_MIGRATION_CONFIGS_READY_VALUE 0xC0FFEUL

/* Migration lifecycle and error status values published through
 * X280_MIGRATION_STATUS_ADDR. */
#define X280_MIGRATION_STATUS_INIT 0xFFFFFFFFu
#define X280_MIGRATION_STATUS_READY 0x10000001u
#define X280_MIGRATION_STATUS_RUNNING 0x10000002u
#define X280_MIGRATION_STATUS_STOPPED 0x10000003u
#define X280_MIGRATION_STATUS_ERROR_RX 0xE0000001u
#define X280_MIGRATION_STATUS_ERROR_TX 0xE0000002u
#define X280_MIGRATION_STATUS_ERROR_BAD_CMD 0xE0000003u
#define X280_MIGRATION_STATUS_ERROR_BAD_BANK 0xE0000004u
#define X280_MIGRATION_STATUS_ERROR_BAD_DRAM_ADDR 0xE0000005u

/* Bench counter offsets within MIGRATION_CONTROL. Each is on its own
 * 64 B cache line so partial-word writes don't false-share with the
 * mailboxes above or with each other. The unallocated range ends at
 * +0x300. */
#define X280_MIGRATION_BENCH_READ_CYCLES_LO_OFFSET 0x300UL
#define X280_MIGRATION_BENCH_READ_CYCLES_HI_OFFSET 0x340UL
#define X280_MIGRATION_BENCH_READ_BYTES_LO_OFFSET 0x380UL
#define X280_MIGRATION_BENCH_READ_BYTES_HI_OFFSET 0x3C0UL
#define X280_MIGRATION_BENCH_READ_CMD_COUNT_OFFSET 0x400UL
#define X280_MIGRATION_BENCH_WRITE_CYCLES_LO_OFFSET 0x440UL
#define X280_MIGRATION_BENCH_WRITE_CYCLES_HI_OFFSET 0x480UL
#define X280_MIGRATION_BENCH_WRITE_BYTES_LO_OFFSET 0x4C0UL
#define X280_MIGRATION_BENCH_WRITE_BYTES_HI_OFFSET 0x500UL
#define X280_MIGRATION_BENCH_WRITE_CMD_COUNT_OFFSET 0x540UL

/* Per-hop split-timer offsets. */
#define X280_MIGRATION_BENCH_READ_DRAM_CYCLES_LO_OFFSET 0x580UL
#define X280_MIGRATION_BENCH_READ_DRAM_CYCLES_HI_OFFSET 0x5C0UL
#define X280_MIGRATION_BENCH_READ_PCIE_CYCLES_LO_OFFSET 0x600UL
#define X280_MIGRATION_BENCH_READ_PCIE_CYCLES_HI_OFFSET 0x640UL
#define X280_MIGRATION_BENCH_WRITE_PCIE_CYCLES_LO_OFFSET 0x680UL
#define X280_MIGRATION_BENCH_WRITE_PCIE_CYCLES_HI_OFFSET 0x6C0UL
#define X280_MIGRATION_BENCH_WRITE_DRAM_CYCLES_LO_OFFSET 0x700UL
#define X280_MIGRATION_BENCH_WRITE_DRAM_CYCLES_HI_OFFSET 0x740UL

/* Socket-page wait counters. */
#define X280_MIGRATION_BENCH_READ_SOCKET_WAIT_CYCLES_LO_OFFSET 0x780UL
#define X280_MIGRATION_BENCH_READ_SOCKET_WAIT_CYCLES_HI_OFFSET 0x7C0UL
#define X280_MIGRATION_BENCH_WRITE_SOCKET_WAIT_CYCLES_LO_OFFSET 0x800UL
#define X280_MIGRATION_BENCH_WRITE_SOCKET_WAIT_CYCLES_HI_OFFSET 0x840UL

/* ------------------------------------------------------------------
 * Multi-hart dispatch mailboxes (inside MIGRATION_CONTROL).
 *
 * Hart 0 (the migration worker proper) parallelizes the slow DRAM-read
 * hop of cmd_handle_read across N harts. Helper harts 1..N-1 spin in
 * multihart_helper_loop polling these mailboxes. Layout (each on its
 * own 64 B cache line so partial-word writes don't false-share with
 * adjacent harts' state):
 *
 *   +0x900 WORK_PRESENT       (hart 0 -> all helpers; WORK_FLAG / 0)
 *   +0x940 HART_DONE[0]       (unused; padded for symmetry)
 *   +0x980 HART_DONE[1]       (helper -> hart 0; DONE_FLAG / 0)
 *   +0x9C0 HART_DONE[2]       (helper -> hart 0; live when
 *                              X280_MULTIHART_NUM_HARTS > 2)
 *   +0xA00 HART_DONE[3]       (helper -> hart 0; live when
 *                              X280_MULTIHART_NUM_HARTS > 2)
 *   +0xA40 WORK_DESC[0]       (uniform hart 0 descriptor slot)
 *   +0xA80 WORK_DESC[1]       (hart 0 -> hart 1's descriptor)
 *   +0xAC0 WORK_DESC[2]       (hart 0 -> hart 2; live when
 *                              X280_MULTIHART_NUM_HARTS > 2)
 *   +0xB00 WORK_DESC[3]       (hart 0 -> hart 3; live when
 *                              X280_MULTIHART_NUM_HARTS > 2)
 *
 * The layout is sized for X280_MULTIHART_MAX_HARTS = 4 even though the
 * current default is 2 harts, so the same mailbox map works whether
 * the firmware is built at 2u or 4u. Each WORK_DESC slot holds an
 * x280_multihart_work struct (32 B) comfortably inside the 64 B
 * cache line.
 * ------------------------------------------------------------------ */
#define X280_MULTIHART_WORK_PRESENT_OFFSET 0x900UL
#define X280_MULTIHART_HART_DONE_BASE_OFFSET 0x940UL
#define X280_MULTIHART_HART_DONE_OFFSET(h) (X280_MULTIHART_HART_DONE_BASE_OFFSET + (uint64_t)(h) * 0x40UL)
#define X280_MULTIHART_WORK_DESC_BASE_OFFSET 0xA40UL
#define X280_MULTIHART_WORK_DESC_OFFSET(h) (X280_MULTIHART_WORK_DESC_BASE_OFFSET + (uint64_t)(h) * 0x40UL)
#define X280_MULTIHART_MAX_HARTS 4u
#define X280_MULTIHART_END_OFFSET (X280_MULTIHART_WORK_DESC_BASE_OFFSET + X280_MULTIHART_MAX_HARTS * 0x40UL)

/* ------------------------------------------------------------------
 * Helper-shutdown handshake (active FW).
 *
 *   +0xE00 X280_HELPER_SHUTDOWN       (active-FW hart 0 -> helpers;
 *                                      u32 X280_HELPER_SHUTDOWN_FLAG;
 *                                      at X280_HELPER_SHUTDOWN_ADDR)
 * ------------------------------------------------------------------ */
#define X280_HELPER_SHUTDOWN_OFFSET 0xE00UL
#define X280_HELPER_SHUTDOWN_ADDR (X280_MIGRATION_CONTROL_ADDR + X280_HELPER_SHUTDOWN_OFFSET)
#define X280_HELPER_SHUTDOWN_FLAG 0x1D1E0DEDU
#define X280_HELPER_SHUTDOWN_END_OFFSET 0xE40UL

/* ------------------------------------------------------------------
 * Per-hart DRAM-read bench counters (X280_BENCHMARK builds only).
 *
 * The aggregate BENCH_READ_DRAM_CYCLES counter above measures hart 0's
 * wall-clock for the *entire* multihart_dispatch_dram_read call (Phases
 * A..F), which conflates per-hart DRAM-load time with the helper-tail
 * barrier wait. To decompose the helper-tail cost from genuine per-hart
 * DRAM throughput we need each hart to publish only the cycles it
 * spent inside x280_dram_read_block_scalar_tlb plus the bytes it
 * actually moved.
 *
 * Each hart owns one 64 B cache line so concurrent writes from
 * different helpers never false-share. Layout inside each hart's slot:
 *   +0   dram_cycles_lo   (u32; cumulative)
 *   +4   dram_cycles_hi   (u32)
 *   +8   dram_bytes_lo    (u32; cumulative)
 *   +12  dram_bytes_hi    (u32)
 *   +16  dram_chunks      (u32; # of x280_dram_read_block_scalar_tlb calls)
 *   +20..+63 padding / reserved
 *
 * Plus one extra cache line at +0x1100 for hart 0 to publish the
 * Phase-E barrier wait (= cycles hart 0 spent stalled after finishing
 * its own chunk, waiting for the slowest helper). That, summed with
 * hart 0's per-chunk dram cycles, gives the full wall-clock view.
 *
 *   +0x1000 PER_HART0_DRAM_*    (hart 0's own run0-lambda chunks)
 *   +0x1040 PER_HART1_DRAM_*
 *   +0x1080 PER_HART2_DRAM_*
 *   +0x10C0 PER_HART3_DRAM_*
 *   +0x1100 PHASE_E_DRAM_CYCLES (hart 0 only; barrier-wait time)
 * ------------------------------------------------------------------ */
#define X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET 0x1000UL
#define X280_MIGRATION_BENCH_PER_HART_STRIDE 0x40UL
#define X280_MIGRATION_BENCH_PER_HART_DRAM_CYCLES_LO_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x00UL)
#define X280_MIGRATION_BENCH_PER_HART_DRAM_CYCLES_HI_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x04UL)
#define X280_MIGRATION_BENCH_PER_HART_DRAM_BYTES_LO_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x08UL)
#define X280_MIGRATION_BENCH_PER_HART_DRAM_BYTES_HI_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x0CUL)
#define X280_MIGRATION_BENCH_PER_HART_DRAM_CHUNKS_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x10UL)
#define X280_MIGRATION_BENCH_PER_HART_END_OFFSET \
    (X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET + X280_MULTIHART_MAX_HARTS * X280_MIGRATION_BENCH_PER_HART_STRIDE)

#define X280_MIGRATION_BENCH_PHASE_E_DRAM_CYCLES_LO_OFFSET 0x1100UL
#define X280_MIGRATION_BENCH_PHASE_E_DRAM_CYCLES_HI_OFFSET 0x1104UL
#define X280_MIGRATION_BENCH_PHASE_E_END_OFFSET 0x1140UL

/* Per-hart PCIe-pull bench counters (X280_BENCHMARK builds only).
 * Mirror of the DRAM-read region above, but for the write path's
 * socket_pull_payload_scalar_at hop. Same one-cache-line-per-hart
 * layout, same in-line offsets. Lives at +0x1200..+0x12FF, with the
 * hart 0 PCIe Phase-E barrier wait at +0x1300. */
#define X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET 0x1200UL
#define X280_MIGRATION_BENCH_PER_HART_PCIE_CYCLES_LO_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x00UL)
#define X280_MIGRATION_BENCH_PER_HART_PCIE_CYCLES_HI_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x04UL)
#define X280_MIGRATION_BENCH_PER_HART_PCIE_BYTES_LO_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x08UL)
#define X280_MIGRATION_BENCH_PER_HART_PCIE_BYTES_HI_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x0CUL)
#define X280_MIGRATION_BENCH_PER_HART_PCIE_CHUNKS_OFFSET(h) \
    (X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET + (uint64_t)(h) * X280_MIGRATION_BENCH_PER_HART_STRIDE + 0x10UL)
#define X280_MIGRATION_BENCH_PER_HART_PCIE_END_OFFSET \
    (X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET + X280_MULTIHART_MAX_HARTS * X280_MIGRATION_BENCH_PER_HART_STRIDE)

#define X280_MIGRATION_BENCH_PHASE_E_PCIE_CYCLES_LO_OFFSET 0x1300UL
#define X280_MIGRATION_BENCH_PHASE_E_PCIE_CYCLES_HI_OFFSET 0x1304UL
#define X280_MIGRATION_BENCH_PHASE_E_PCIE_END_OFFSET 0x1340UL

/* ------------------------------------------------------------------
 * COPY-path bench counters (X280_BENCHMARK builds only).
 *
 * Symmetric with the READ/WRITE aggregate counters at +0x300..+0x540
 * and the per-hop split timers at +0x580..+0x740, but COPY has no
 * host data socket so it gets no socket_wait counter. It occupies the
 * disjoint region after PER_HART_PCIE_END (+0x1340).
 *
 *   +0x1400 BENCH_COPY_CYCLES_LO          (u32; cumulative mtime)
 *   +0x1440 BENCH_COPY_CYCLES_HI          (u32)
 *   +0x1480 BENCH_COPY_BYTES_LO           (u32; cumulative num_bytes)
 *   +0x14C0 BENCH_COPY_BYTES_HI           (u32)
 *   +0x1500 BENCH_COPY_CMD_COUNT          (u32; cumulative cmd count)
 *   +0x1540 BENCH_COPY_DRAM_READ_CYCLES_LO  (u32; src DRAM-read hop;
 *                                            zero in DMA_NO_STAGING)
 *   +0x1580 BENCH_COPY_DRAM_READ_CYCLES_HI  (u32)
 *   +0x15C0 BENCH_COPY_DRAM_WRITE_CYCLES_LO (u32; dst DRAM-write hop;
 *                                            zero in DMA_NO_STAGING)
 *   +0x1600 BENCH_COPY_DRAM_WRITE_CYCLES_HI (u32)
 *
 * Note: COPY's SCALAR-mode src DRAM-read hop shares the PER_HART_DRAM
 * mailboxes at +0x1000..+0x10FF with READ's per-hart
 * decomposition (cmd_handle_copy calls multihart_dispatch_dram_read,
 * same helper as cmd_handle_read). In a copy-only benchmark those
 * mailboxes attribute entirely to COPY; in a mixed READ+COPY run
 * they conflate the two and the per-hart attribution should be
 * interpreted accordingly.
 * ------------------------------------------------------------------ */
#define X280_MIGRATION_BENCH_COPY_CYCLES_LO_OFFSET 0x1400UL
#define X280_MIGRATION_BENCH_COPY_CYCLES_HI_OFFSET 0x1440UL
#define X280_MIGRATION_BENCH_COPY_BYTES_LO_OFFSET 0x1480UL
#define X280_MIGRATION_BENCH_COPY_BYTES_HI_OFFSET 0x14C0UL
#define X280_MIGRATION_BENCH_COPY_CMD_COUNT_OFFSET 0x1500UL
#define X280_MIGRATION_BENCH_COPY_DRAM_READ_CYCLES_LO_OFFSET 0x1540UL
#define X280_MIGRATION_BENCH_COPY_DRAM_READ_CYCLES_HI_OFFSET 0x1580UL
#define X280_MIGRATION_BENCH_COPY_DRAM_WRITE_CYCLES_LO_OFFSET 0x15C0UL
#define X280_MIGRATION_BENCH_COPY_DRAM_WRITE_CYCLES_HI_OFFSET 0x1600UL
#define X280_MIGRATION_BENCH_COPY_END_OFFSET 0x1640UL

/* ------------------------------------------------------------------
 * DRAM bank table (region 2).
 *
 * Host pre-writes a packed array of (noc_xy_enc, base_addr, bank_size,
 * _pad) records, one per DRAM bank the firmware needs to address.
 * Indexed by the bank_id field of an incoming x280_migration_cmd_t.
 *
 * Capacity X280_DRAM_BANK_TABLE_MAX_BANKS * 16 B = 1 KiB max; the
 * 4 KiB reservation leaves room for an eventual per-entry size /
 * channel field. Blackhole has ~12 DRAM banks; sizing at 64 gives
 * headroom for board variants.
 * ------------------------------------------------------------------ */
#define X280_DRAM_BANK_TABLE_OFFSET 0x00132000UL
#define X280_DRAM_BANK_TABLE_SIZE 0x00001000UL /* 4 KiB reserved */
#define X280_DRAM_BANK_TABLE_ADDR (X280_LIM_BASE + X280_DRAM_BANK_TABLE_OFFSET)
#define X280_DRAM_BANK_TABLE_MAX_BANKS 64u

/* ------------------------------------------------------------------
 * Socket slot table + per-socket configs (region 3).
 *
 * Hard-capped at X280_MAX_SOCKETS = 5 sockets. Each socket has:
 *   - one 8-byte slot in the slot table at +0x00..+0x28
 *     (config_addr, fifo_addr)
 *   - one 128-byte config blob slot starting at +0x40
 *     (5 * 128 B = 640 B; cache-line aligned)
 *
 * The host fills the slot table; the firmware reads each role's
 * slot to find that role's cfg / fifo addresses.
 * ------------------------------------------------------------------ */
#define X280_MAX_SOCKETS 5u
#define X280_SOCKET_CONFIG_SIZE 128u   /* per-socket cfg blob */
#define X280_SOCKET_SLOT_ENTRY_SIZE 8u /* {config_addr, fifo_addr} */

#define X280_SOCKET_SLOT_TABLE_OFFSET 0x00133000UL
#define X280_SOCKET_SLOT_TABLE_ADDR (X280_LIM_BASE + X280_SOCKET_SLOT_TABLE_OFFSET)
#define X280_SOCKET_SLOT_TABLE_SIZE (X280_MAX_SOCKETS * X280_SOCKET_SLOT_ENTRY_SIZE) /* 40 B */

/* Per-socket config blobs follow the slot table at +0x40 (rounded up
 * to a cache line, leaves the 40-byte slot table on its own line). */
#define X280_SOCKET_CONFIGS_OFFSET (X280_SOCKET_SLOT_TABLE_OFFSET + 0x40UL)
#define X280_SOCKET_CONFIGS_ADDR (X280_LIM_BASE + X280_SOCKET_CONFIGS_OFFSET)
#define X280_SOCKET_CONFIGS_SIZE (X280_MAX_SOCKETS * X280_SOCKET_CONFIG_SIZE) /* 640 B */

/* Helper: address of the i-th per-socket config slot. The host's
 * X280SocketAllocator hands out these addresses in order; the firmware
 * never computes them directly, it reads slot[i].config_addr from the
 * slot table. */
#define X280_SOCKET_CONFIG_SLOT_ADDR(i) (X280_SOCKET_CONFIGS_ADDR + (uint32_t)(i) * X280_SOCKET_CONFIG_SIZE)

/* Logical socket roles. Slots are indexed by role at firmware boot.
 * Add new roles only at the end and bump X280_MAX_SOCKETS in
 * lock-step. */
typedef enum {
    X280_SOCKET_ROLE_H2D_CMD = 0,        /* HOST_PUSH command ring */
    X280_SOCKET_ROLE_D2H = 1,            /* D2H data ring */
    X280_SOCKET_ROLE_H2D_DATA = 2,       /* DEVICE_PULL (or HOST_PUSH) data */
    X280_SOCKET_ROLE_D2H_COMPLETION = 3, /* D2H migration-completion ring (reserved) */
    X280_SOCKET_ROLE_SPARE = 4,          /* reserved for future use */
} x280_socket_role_t;

/* Slot-table entry. Plain POD so the host can pack identical wire
 * bytes from Python via struct.pack("<II", ...). */
typedef struct {
    uint32_t config_addr; /* LIM addr of cfg blob (zero if unused) */
    uint32_t fifo_addr;   /* LIM addr of H2D HOST_PUSH ring base
                             (zero for DEVICE_PULL anchor or D2H) */
} x280_socket_slot_t;

static_assert(
    sizeof(x280_socket_slot_t) == X280_SOCKET_SLOT_ENTRY_SIZE, "x280_socket_slot_t must be 8 bytes wire-compatible.");

/* ------------------------------------------------------------------
 * Staging buffer (region 4).
 *
 * LIM scratch the migration worker uses to stage a single max-page-
 * size chunk between the DRAM NOC and the data ring. The firmware
 * can't issue a single NOC-to-PCIe DMA in one step:
 *   - READ:  noc_read DRAM -> staging   (NOC), then
 *            socket_write_payload staging -> pinned host (PCIe TLB)
 *   - WRITE: socket_pull_payload pinned host -> staging (PCIe TLB),
 *            then noc_write staging -> DRAM (NOC)
 * Default 64 KiB. Override before include if the migration worker is
 * built with a larger max data_page_size.
 * ------------------------------------------------------------------ */
#define X280_STAGING_OFFSET 0x00134000UL
#ifndef X280_STAGING_SIZE
#define X280_STAGING_SIZE 0x00010000UL /* 64 KiB */
#endif
#define X280_STAGING_ADDR (X280_LIM_BASE + X280_STAGING_OFFSET)

/* ------------------------------------------------------------------
 * H2D-FIFO bump pool (region 5, highest offset).
 *
 * Single contiguous LIM region from which the host's allocator
 * bump-allocates per-socket H2D HOST_PUSH ring buffers. H2D
 * DEVICE_PULL and D2H sockets do not claim bytes here: the former's
 * payload lives in pinned host RAM (DEVICE_PULL just stores a logical
 * anchor in slot[i].fifo_addr) and the latter's FIFO lives entirely
 * in pinned host RAM.
 * ------------------------------------------------------------------ */
#define X280_H2D_FIFO_POOL_OFFSET 0x00144000UL
#ifndef X280_H2D_FIFO_POOL_SIZE
#define X280_H2D_FIFO_POOL_SIZE 0x00010000UL /* 64 KiB */
#endif
#define X280_H2D_FIFO_POOL_ADDR (X280_LIM_BASE + X280_H2D_FIFO_POOL_OFFSET)

/* End of the highest reserved offset (for compile-time sanity checks
 * against LIM size and against the stack). Update whenever a new region
 * is added above. */
#define X280_SOCKET_HIGHEST_END (X280_H2D_FIFO_POOL_OFFSET + X280_H2D_FIFO_POOL_SIZE)

/* ------------------------------------------------------------------
 * TLB index reservations for socket use.
 * ------------------------------------------------------------------
 * x280.h documents indices in use by other phases:
 *   16-23  Phase 4 NOC write
 *   24-28  Phase 5 NOC read / ordering
 *   32-33  Phase 6 integration
 *   34-36  Phase 8 bench
 *   37-44  Phase 9 experiments
 *   45-46  Phase 9b memport
 *   50     PCIe NoC-write spike
 * Sockets reserve a small block starting at 60 to leave headroom for
 * future probe / experiment indices. */
#define X280_SOCKET_TLB_D2H_DATA 60u   /* device -> pinned data FIFO   */
#define X280_SOCKET_TLB_D2H_NOTIFY 61u /* device -> pinned bytes_sent  */
#define X280_SOCKET_TLB_H2D_CMD_NOTIFY  \
    62u /* device -> pinned bytes_acked \
           (h2d-cmd socket) */
/* Simple H2D receivers use the command-notify TLB by default. */
#define X280_SOCKET_TLB_H2D_NOTIFY X280_SOCKET_TLB_H2D_CMD_NOTIFY
/* Migration-worker data-ring additions: */
#define X280_SOCKET_TLB_H2D_DATA_PULL   \
    63u /* device <- pinned data FIFO   \
           (DEVICE_PULL read window for \
           the h2d-data socket) */
#define X280_SOCKET_TLB_H2D_DATA_NOTIFY \
    64u /* device -> pinned bytes_acked \
           (h2d-data socket) */
#define X280_SOCKET_TLB_DRAM_RW         \
    65u /* device <-> DRAM (staging     \
           chunk path; reprogrammed per \
           bank within a command) */
/* Per-hart DRAM-read TLB slots used by the multi-hart parallel
 * dram_read_block helper. Hart 0, dram_write_block, and DMA paths share
 * slot 65. Helper harts 1..3 own one slot each so concurrent reads on
 * the same DRAM bank don't race on TLB programming. */
#define X280_SOCKET_TLB_DRAM_RW_HART0 65u
#define X280_SOCKET_TLB_DRAM_RW_HART1 66u
#define X280_SOCKET_TLB_DRAM_RW_HART2 67u
#define X280_SOCKET_TLB_DRAM_RW_HART3 68u
#define X280_SOCKET_TLB_DRAM_RW_PER_HART(h) (X280_SOCKET_TLB_DRAM_RW_HART0 + (uint32_t)(h))

/* Per-hart PCIe-pull TLB slots for the WRITE-path multi-hart split.
 * Hart 0 keeps slot 63 (alias of X280_SOCKET_TLB_H2D_DATA_PULL) so
 * the single-hart socket_pull_payload_scalar fallback continues to
 * work unmodified; helper harts 1..3 each own a distinct slot so
 * concurrent pulls don't race on TLB programming. Slots 69-71 carry
 * the helper-side slots (slot 63 doubles as HART0). */
#define X280_SOCKET_TLB_H2D_DATA_PULL_HART0 63u
#define X280_SOCKET_TLB_H2D_DATA_PULL_HART1 69u
#define X280_SOCKET_TLB_H2D_DATA_PULL_HART2 70u
#define X280_SOCKET_TLB_H2D_DATA_PULL_HART3 71u
#define X280_SOCKET_TLB_H2D_DATA_PULL_PER_HART(h) \
    (((h) == 0u) ? X280_SOCKET_TLB_H2D_DATA_PULL_HART0 : (X280_SOCKET_TLB_H2D_DATA_PULL_HART1 + ((uint32_t)(h) - 1u)))
/* Index 72 onward reserved for future socket needs (multi-receiver
 * D2H, D2D, multi-page chunked writes, etc.). */

/* ------------------------------------------------------------------
 * Compile-time guards: each region must fit before the next, and the
 * top of the FIFO pool must stay below the firmware stack. Update
 * whenever a new region is added above.
 * ------------------------------------------------------------------ */
static_assert(
    X280_ACTIVE_FW_REGION_END <= X280_LIM_BASE + X280_SPIKE_MBOX_OFFSET,
    "X280 active-firmware region overlaps the spike mailbox region.");
static_assert(
    X280_SPIKE_MBOX_OFFSET + X280_SPIKE_MBOX_SIZE <= X280_MIGRATION_CONTROL_OFFSET,
    "X280 spike mailbox region overlaps the migration-control region.");
static_assert(
    X280_MIGRATION_CONTROL_OFFSET + X280_MIGRATION_CONTROL_SIZE <= X280_DRAM_BANK_TABLE_OFFSET,
    "Migration-control region overflows into the DRAM bank table.");
static_assert(
    X280_MIGRATION_BENCH_WRITE_SOCKET_WAIT_CYCLES_HI_OFFSET + 0x40UL <= X280_MIGRATION_CONTROL_SIZE,
    "Migration bench counters overflow the MIGRATION_CONTROL reservation.");

static_assert(
    X280_MULTIHART_END_OFFSET <= X280_MIGRATION_CONTROL_SIZE,
    "Multi-hart dispatch region overflows the MIGRATION_CONTROL reservation.");

static_assert(
    X280_HELPER_SHUTDOWN_OFFSET >= X280_MULTIHART_END_OFFSET,
    "Helper shutdown region overlaps the multihart dispatch region.");
static_assert(
    X280_HELPER_SHUTDOWN_END_OFFSET <= X280_MIGRATION_CONTROL_SIZE,
    "Helper shutdown region overflows the MIGRATION_CONTROL reservation.");

static_assert(
    X280_MIGRATION_BENCH_PER_HART_BASE_OFFSET >= X280_HELPER_SHUTDOWN_END_OFFSET,
    "Per-hart bench region overlaps the helper-shutdown region.");
static_assert(
    X280_MIGRATION_BENCH_PHASE_E_END_OFFSET <= X280_MIGRATION_CONTROL_SIZE,
    "Per-hart bench region overflows the MIGRATION_CONTROL reservation.");
static_assert(
    X280_MIGRATION_BENCH_PER_HART_PCIE_BASE_OFFSET >= X280_MIGRATION_BENCH_PHASE_E_END_OFFSET,
    "Per-hart PCIe bench region overlaps the DRAM Phase-E region.");
static_assert(
    X280_MIGRATION_BENCH_PHASE_E_PCIE_END_OFFSET <= X280_MIGRATION_CONTROL_SIZE,
    "Per-hart PCIe bench region overflows the MIGRATION_CONTROL reservation.");

static_assert(
    X280_MIGRATION_BENCH_COPY_CYCLES_LO_OFFSET >= X280_MIGRATION_BENCH_PHASE_E_PCIE_END_OFFSET,
    "COPY bench region overlaps the per-hart PCIe bench region.");
static_assert(
    X280_MIGRATION_BENCH_COPY_END_OFFSET <= X280_MIGRATION_CONTROL_SIZE,
    "COPY bench region overflows the MIGRATION_CONTROL reservation.");
static_assert(
    X280_HART_PROTOCOL_BASE_OFFSET >= X280_MIGRATION_BENCH_COPY_END_OFFSET,
    "Per-hart lifecycle records overlap COPY bench counters.");
static_assert(
    X280_HART_PROTOCOL_END_OFFSET <= X280_MIGRATION_CONTROL_SIZE,
    "Per-hart lifecycle records overflow MIGRATION_CONTROL.");
static_assert(
    X280_HART_PROTOCOL_BASE == X280_MIGRATION_CONTROL_ADDR + X280_HART_PROTOCOL_BASE_OFFSET,
    "Per-hart lifecycle base disagrees with the migration-control layout.");

/* Offsets +0x180..+0x2FF are unallocated. */
static_assert(
    X280_MIGRATION_BENCH_READ_CYCLES_LO_OFFSET >= 0x300UL,
    "Bench counters overlap the unallocated range at +0x180..+0x2FF.");
static_assert(
    X280_DRAM_BANK_TABLE_OFFSET + X280_DRAM_BANK_TABLE_SIZE <= X280_SOCKET_SLOT_TABLE_OFFSET,
    "DRAM bank table overflows into the socket slot table region.");
static_assert(
    X280_SOCKET_SLOT_TABLE_SIZE <= 0x40UL, "Socket slot table must fit in the 0x40 B reserved before the configs.");
static_assert(
    X280_SOCKET_CONFIGS_OFFSET + X280_SOCKET_CONFIGS_SIZE <= X280_STAGING_OFFSET,
    "Socket configs region overflows into the staging buffer.");
static_assert(
    X280_STAGING_OFFSET + X280_STAGING_SIZE <= X280_H2D_FIFO_POOL_OFFSET,
    "X280_STAGING_SIZE overflows into the H2D-FIFO bump pool.");
static_assert(
    X280_SOCKET_HIGHEST_END <= 0x001D8000UL, "X280 socket layout overflows LIM (must stay below stack at 0x081D8000).");
static_assert(
    X280_LIM_BASE + X280_SOCKET_HIGHEST_END <= X280_LIM_END, "X280 socket layout overflows the canonical LIM bounds.");

#endif /* SOCKET_LAYOUT_H */
