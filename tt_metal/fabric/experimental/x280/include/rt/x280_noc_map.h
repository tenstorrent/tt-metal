// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_noc_map.h -- CPU NOC TLB slot reservations.
 *
 * The 224 2 MiB TLB entries are per-tile state, and only one firmware image is
 * resident at a time, so the ranges below overlap harmlessly. The invariant is
 * per-image: adding a slot means checking what else that SAME image programs,
 * not looking for a globally free index.
 *
 * INDICES 0..8 ARE NOT RELIABLE. ARC overwrites them after the firmware
 * configures them, so a configure/read-back check there fails; anything that
 * verifies its TLB configuration needs 16 or above. The DMA engine is the
 * exception: its NIU master only honours the window programmed right before
 * the kick, and higher indices are empirically flaky for it, so a DMA user
 * reprograms slots 0 and 1 immediately before every kick.
 */
#ifndef X280_RT_NOC_MAP_H
#define X280_RT_NOC_MAP_H

/* 2 MiB CPU NOC TLB slots
 *
 *   slots     owner                      declared in
 *   16..155   packet-injection credits   X280_TLB_PACKET_INJECTION_CREDIT_BASE, below
 *   16..19    eth_credit forwarders      X280_TLB_ETH_CREDIT_BASE, below
 *   34        firmware posted write      X280_TLB_BENCH_WRITE, below
 *   120..127  paired-case outbound       X280_TLB_PAIRED_BASE, below
 *   156       packet-injection timer      X280_TLB_PACKET_INJECTION_TIMER, below
 *   157..160  core-sync barrier          X280_TLB_CORE_SYNC_BASE, below
 *   192..223  host-programmed            X280_TLB_HOST_BASE, below
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

/* A firmware-owned posted-write window, outside the host block. */
#define X280_TLB_BENCH_WRITE 34

#endif /* X280_RT_NOC_MAP_H */
