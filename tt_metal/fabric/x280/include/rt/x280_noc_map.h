// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_noc_map.h -- CPU NOC TLB slot reservations and destination tiles.
 *
 * The 224 2 MiB TLB entries are per-tile state, and only one firmware image is
 * resident at a time, so the ranges below overlap harmlessly. The invariant is
 * per-image: adding a slot means checking what else that SAME image programs,
 * not looking for a globally free index.
 *
 * INDICES 0..8 ARE NOT RELIABLE. ARC overwrites them after the firmware
 * configures them, so a configure/read-back check there fails; anything that
 * verifies its TLB configuration needs 16 or above. Three consumers sit in the
 * range regardless and tolerate it:
 *
 *   the DMA engine (0, 1)        reprograms the slot immediately before every
 *                                kick. The DMA NIU master only honours the
 *                                window programmed right before the kick, and
 *                                higher indices are empirically flaky here.
 *   multi-window bench (0..7)    never reads a TLB config back
 *   scatter bench (0..31)        never reads a TLB config back
 *
 * For those three the slot index is part of what is measured, so moving them
 * higher changes their results rather than fixing anything.
 */
#ifndef X280_RT_NOC_MAP_H
#define X280_RT_NOC_MAP_H

/* 2 MiB CPU NOC TLB slots
 *
 *   slots     owner                      declared in
 *   0, 1      DMA engine src/dst         the DMA engine image
 *   0..7      multi-window bench         its image
 *   0..31     scatter, per-Tensix        its image
 *   16..23    NOC write/read tests       X280_TLB_NOC_TEST_BASE, below
 *   16..155   packet-injection credits   X280_TLB_PACKET_INJECTION_CREDIT_BASE, below
 *   16..19    eth_credit forwarders      X280_TLB_ETH_CREDIT_BASE, below
 *   24..28    NOC read + ordering        X280_TLB_NOC_PHASE5_BASE, below
 *   32, 33    integration write/read     X280_TLB_INT_*, below
 *   34..38    benchmark suite            X280_TLB_BENCH_*, below
 *   40..43    per-hart DDR bandwidth     X280_TLB_DDR_MULTIHART_BASE, below
 *   47, 48    DDR fill / read windows    X280_TLB_DDR_FILL / _READ, below
 *   50        PCIe pinned-host write     X280_TLB_PCIE_SPIKE, below
 *   50..81    scatter, per-core          its image
 *   60..71    socket data/notify/DRAM    socket_layout.h (X280_SOCKET_TLB_*)
 *   100..103  scatter, per-hart          its image
 *   100..107  stress, per-hart wr/rd     the stress image
 *   120..127  paired-case outbound       X280_TLB_PAIRED_BASE, below
 *   156       packet-injection timer      X280_TLB_PACKET_INJECTION_TIMER, below
 *   157..160  core-sync barrier          X280_TLB_CORE_SYNC_BASE, below
 *   192..223  host-programmed            X280_TLB_HOST_BASE, below
 *
 * The socket block lives in socket_layout.h because those indices are part of
 * the host-visible layout contract that file defines.
 */

/* Windows the host may program on a tile's behalf, so a case can be handed a
 * ready window instead of a kernel hard-coding a peer coordinate.
 *
 * At the top of the range because every firmware block above allocates upward
 * from 0: a host block anywhere lower would be overrun by the next image that
 * needs a few more slots, and the collision is silent -- both sides write the
 * same config register and the loser's window simply points somewhere else.
 *
 * Firmware must not program these, and the host must not program any other. */
#define X280_TLB_HOST_BASE 192
#define X280_TLB_HOST_COUNT 32

/* Outbound windows an image may program when it is one half of a paired case,
 * reaching a core another participant owns. A range and not one slot: two paired
 * cases in the tree would otherwise share a config register, and that collision
 * is silent -- the loser's window resolves, just somewhere else.
 *
 * Which slot in the range a case takes is the host's to choose and to tell the
 * image; the image only checks that what it was told falls inside. Clear of
 * every firmware block above and below X280_TLB_HOST_BASE, and well above 8, so
 * an image can verify its own programming by read-back without ARC rewriting it
 * underneath. */
#define X280_TLB_PAIRED_BASE 120u
#define X280_TLB_PAIRED_COUNT 8u

#if !defined(__ASSEMBLER__)
#include <stdbool.h>
#include <stdint.h>

/** Whether `slot` lies in the paired-case range. */
static inline bool x280_tlb_paired_slot(uint32_t slot) {
    return slot >= X280_TLB_PAIRED_BASE && slot < X280_TLB_PAIRED_BASE + X280_TLB_PAIRED_COUNT;
}
#endif

/* Outbound windows reserved for core-synchronization barrier operations.
 *
 * One per hart, because the four harts of a tile share this configuration block
 * and any of them may be a barrier participant at the same time. A slot per
 * worker is not what is needed and would not fit: a participant retires its
 * window between writes, so only its own concurrent use has to be separated.
 *
 * At 157 the block sits clear of every firmware block that allocates upward
 * from 0 and of the paired-case range below it, and clear of the host block
 * above; slot 157 is hardware-qualified for sequential singleton and multi-group
 * release. The remaining slots are software ownership only.
 *
 * Which slot a participant takes is the host's to choose and to state in the
 * config; the transport checks only that it falls inside this block. */
#define X280_TLB_CORE_SYNC_BASE 157u
#define X280_TLB_CORE_SYNC_COUNT 4u

/* Packet-injection TLB reservations: 140 credit windows and 1 timer window. */
#define X280_TLB_PACKET_INJECTION_CREDIT_BASE 16u
#define X280_TLB_PACKET_INJECTION_CREDIT_COUNT 140u
#define X280_TLB_PACKET_INJECTION_TIMER 156u

#if defined(__cplusplus)
static_assert(
    X280_TLB_PACKET_INJECTION_CREDIT_BASE + X280_TLB_PACKET_INJECTION_CREDIT_COUNT <= X280_TLB_PACKET_INJECTION_TIMER,
    "packet injection credit interval must not overlap timer slot");
static_assert(
    X280_TLB_PACKET_INJECTION_TIMER < X280_TLB_CORE_SYNC_BASE,
    "packet injection timer slot must stay below the core-sync block");
#elif !defined(__ASSEMBLER__)
_Static_assert(
    X280_TLB_PACKET_INJECTION_CREDIT_BASE + X280_TLB_PACKET_INJECTION_CREDIT_COUNT <= X280_TLB_PACKET_INJECTION_TIMER,
    "packet injection credit interval must not overlap timer slot");
_Static_assert(
    X280_TLB_PACKET_INJECTION_TIMER < X280_TLB_CORE_SYNC_BASE,
    "packet injection timer slot must stay below the core-sync block");
#endif

/* The relations in this file a compiler can check. Every other reservation
 * above overlaps by design -- one image is resident at a time -- but the host
 * block is concurrent with firmware, so an overlap there is a live collision
 * rather than a harmless one. */
#if defined(__cplusplus)
static_assert(
    X280_TLB_PAIRED_BASE + X280_TLB_PAIRED_COUNT <= X280_TLB_CORE_SYNC_BASE,
    "paired-case TLB range must stay below the core-sync block");
static_assert(
    X280_TLB_CORE_SYNC_BASE + X280_TLB_CORE_SYNC_COUNT <= X280_TLB_HOST_BASE,
    "core-sync TLB range must stay below the host block");
#elif !defined(__ASSEMBLER__)
_Static_assert(
    X280_TLB_PAIRED_BASE + X280_TLB_PAIRED_COUNT <= X280_TLB_CORE_SYNC_BASE,
    "paired-case TLB range must stay below the core-sync block");
_Static_assert(
    X280_TLB_CORE_SYNC_BASE + X280_TLB_CORE_SYNC_COUNT <= X280_TLB_HOST_BASE,
    "core-sync TLB range must stay below the host block");
#endif

/* eth_credit forwarder windows. Forwarder hart p owns PER_HART slots from
 * base + p * PER_HART: one onto its Ethernet tile, which carries W1, W2 and R1
 * because E0 and E1 share that tile's L1. */
#define X280_TLB_ETH_CREDIT_BASE 16u
#define X280_TLB_ETH_CREDIT_PER_HART 1u
#define X280_TLB_ETH_CREDIT_COUNT 4u

#if defined(__cplusplus)
static_assert(
    X280_TLB_ETH_CREDIT_BASE + X280_TLB_ETH_CREDIT_COUNT <= X280_TLB_CORE_SYNC_BASE,
    "eth_credit forwarder windows must stay below the core-sync block its director uses");
#elif !defined(__ASSEMBLER__)
_Static_assert(
    X280_TLB_ETH_CREDIT_BASE + X280_TLB_ETH_CREDIT_COUNT <= X280_TLB_CORE_SYNC_BASE,
    "eth_credit forwarder windows must stay below the core-sync block its director uses");
#endif

/* NOC write tests: four targets x 2 modes (posted, non-posted), base + idx * 2. */
#define X280_TLB_NOC_TEST_BASE 16

/* NOC read tests: roundtrip and ordering, base + 0..4. */
#define X280_TLB_NOC_PHASE5_BASE 24

/* Integration tests */
#define X280_TLB_INT_WRITE 32 /* posted     */
#define X280_TLB_INT_READ 33  /* non-posted */

/* NOC benchmarks */
#define X280_TLB_BENCH_WRITE 34       /* posted write                */
#define X280_TLB_BENCH_READ 35        /* non-posted read             */
#define X280_TLB_BENCH_WRITE2 36      /* dual-window interleave      */
#define X280_TLB_BENCH_READ_SCALAR 37 /* non-posted, scalar read BW  */
#define X280_TLB_BENCH_READ_VECTOR 38 /* non-posted, vector read BW  */

/* Per-hart DDR bandwidth: one slot per hart, base + hartid. */
#define X280_TLB_DDR_MULTIHART_BASE 40

/* Bandwidth-matrix and concurrent-DDR tests: the DDR-side windows
 * used to prime and then read back a region over the NOC. */
#define X280_TLB_DDR_FILL 47
#define X280_TLB_DDR_READ 48

/* PCIe NoC-write spike: window onto pinned host RAM via the PCIe tile. */
#define X280_TLB_PCIE_SPIKE 50u

/* NOC destination tiles, as (x, y); pack with x280_noc_xy_pack from x280_noc_coord.h.
 *
 * Only DRAM (9,3) and (9,8) are reachable from L2CPU0 -- every other bank
 * hangs the access (BUG-3). Tensix y=0 is router-only and y=1 is the ethernet
 * row, so the first worker on any column is at y=2.
 */

#define X280_NOC_DRAM_L2CPU0_X 9 /* paired DRAM bank, BUG-3 safe */
#define X280_NOC_DRAM_L2CPU0_Y 3
#define X280_NOC_DRAM_L2CPU1_X 9 /* peer DRAM bank,   BUG-3 safe */
#define X280_NOC_DRAM_L2CPU1_Y 8

#define X280_NOC_TENSIX_A_X 1 /* Tensix worker (1,11)  */
#define X280_NOC_TENSIX_A_Y 11
#define X280_NOC_TENSIX_B_X 16 /* Tensix worker (16,2)  */
#define X280_NOC_TENSIX_B_Y 2

#endif /* X280_RT_NOC_MAP_H */
