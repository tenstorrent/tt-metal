// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the ERISC load-overlap microbenchmark's host test and kernel.

#pragma once

#include <cstddef>
#include <cstdint>

namespace erisc_load_overlap {

// Patterns. "Use" = an add that consumes the loaded register; filler = K independent addi instructions.
// Reads: S, S2 = stream registers; L, L2 = L1 words; D = a word in local data memory (where the stack lives).
enum Pattern : uint32_t {
    BASE2,           // K filler + 2 adds, no loads
    DEP_STREAM,      // S used at once, K
    DEP_L1,          // L used at once, K
    DEP_BOTH,        // S used at once, L used at once, K (today's empty-channel visit)
    PRE_STREAM,      // S, K, use
    PRE_L1,          // L, K, use
    PRE_BOTH,        // S + L, K, use both (the proposal)
    PRE_BOTH_REV,    // L + S, K, use both
    PRE_STREAM_LDM,  // S + D, K, use both
    PRE_STREAM2,     // S + S2, K, use both
    PRE_L1_LDM,      // L + D, K, use both
    PRE_L1_2,        // L + L2, K, use both
    PRE_STREAM_X0,   // S + a load from D into x0 (discarded), K, use S
    BASE3,           // K filler + 3 adds, no loads
    DEP_LDM,         // D used at once, K
    DEP_STREAM2,     // S2 used at once, K
    PRE_BOTH_LDM,    // S + L, K/2, D used at once, K/2, use both (a spill reload while the reads are in flight)
    PRE_BOTH_MMIO,   // S + L, K/2, S2 used at once, K/2, use both (e.g. the TX queue busy check)
    PRE_BOTH_TAIL,   // S + L + D, K, use all three
    PRE_S2L,         // S + S2 + L, K, use all three
    BASE4,           // K filler + 4 adds, no loads
    PRE_S4,          // four stream registers back to back, K, use all four (every channel's packet check at once)
    // Other MMIO registers the router reads (RESULTS §17 measured stream registers only). T = TX queue 1's command
    // register, Q = its status register (the busy check reads T, discards it, then reads Q: the BH-55 workaround),
    // N = a NoC write-request status register (the receiver's "write sent" check), C = a NoC command buffer's ready
    // register. Nothing is sent, so the TX queue and the NoC stay idle.
    BASE5,         // K filler + 5 adds, no loads
    PRE_S4_L,      // S, S2, S3, S4 + L, K, use all five (E0's grouped read with the teardown semaphore folded in)
    DEP_TXQ,       // T used at once, K
    DEP_TXQ_BUSY,  // T (discarded) + Q used at once, K (today's busy check)
    DEP_NIU,       // N used at once, K
    DEP_CMDBUF,    // C used at once, K
    PRE_TXQ_BUSY,  // T (discarded) + Q, K, use Q (the busy check issued early)
    PRE_S_TXQ,     // S + T (discarded) + Q, K, use S and Q (arrival count and busy check together)
    PRE_S_NIU,     // S + N, K, use both (arrival count and write-sent check together)
    PRE_E1_GROUP,  // S + T (discarded) + Q + N, K, use S, Q, N (all of E1's per-iteration register reads)
    // The router's compiled busy check loads T and Q into the same register; these variants do the same.
    DEP_TXQ_BUSY_WAW,  // T + Q into one register, Q used at once, K
    PRE_E1_GROUP_WAW,  // S + T, Q into one register + N, K, use S, Q, N
    PRE_S4_TXQ,        // S, S2, S3, S4 + T, Q into one register, K, use all five (E0's grouped read + its busy check)
    // Parked send cursors (V4_STATE.md, 2026-10-30): local-memory words loaded with the grouped stream reads, and the
    // send's header read through a slot address held in a register or loaded from local memory. D, D2, D3 = three
    // local-memory words; DS = a local-memory word holding the address of the L1 word L (a parked slot).
    BASE7,           // K filler + 7 adds, no loads
    PRE_S4_D,        // S, S2, S3, S4, then D, K, use all five
    PRE_D_S4,        // D, then S, S2, S3, S4, K, use all five
    PRE_S4_D3,       // S, S2, S3, S4, then D, D2, D3, K, use all seven
    PRE_D3_S4,       // D, D2, D3, then S, S2, S3, S4, K, use all seven
    PRE_S4_HDR,      // S..S4, K, use them, then L (the header, slot in a register) used at once
    PRE_S4_DS_HDR,   // S..S4 + DS (the slot, loaded with the group), K, use S..S4, then L through DS used at once
    DEP_DS_HDR,      // DS, then L through it used at once (slot loaded right before the header), K
    // Several packets per visit (fabric_exp/consolidation/V4_SEND_BATCH.md, 2026-11-01). Every block starts as a sender
    // pass does: the four stream registers read as a group, 6 filler, all four used (the count read). Then header loads:
    // H0..H3 = the payload-size halfword (offset 40) of four ring slots 4448 B apart (a 4352 B packet's slot); DS, DS2 =
    // local-memory words holding H0's and H1's slot addresses (a forwarding channel's parked slot). Cost = cycles per
    // block minus HB_BASE (same shape, no header loads; 4 uses).
    HB_BASE,   // count read; K; 4 uses of filler
    HB_H1,     // count read; H0, K, use
    HB_H2,     // count read; H0 H1 issued together, K, use both
    HB_H3,     // count read; H0 H1 H2 together, K, use all
    HB_H4,     // count read; H0..H3 together, K, use all
    HB_H2_SEQ, // count read; H0 used at once, H1 used at once, K
    HB_H4_SEQ, // count read; H0..H3 each used at once, K
    HB_C1,     // count read; DS, then H0 through it, K, use (one forwarding channel, slot parked)
    HB_C2,     // count read; DS DS2 together, then H0 H1 through them together, K, use both
    HB_C2_SEQ, // count read; DS, H0 through it used at once; DS2, H1 through it used at once; K
    // One-ahead header prefetch behind the send's TX-queue writes. W = the seven TX-queue register writes of a send (data
    // command 4, packets-received update 3), here to TX queue 1's staging registers only (start address, destination,
    // size, remote register data: never the command register, so nothing is sent). Cost against HW_BASE.
    HW_BASE,   // count read; 7 writes, K, 2 uses of filler
    HW_PF,     // count read; H1, 7 writes, K, use (the next header issued before the writes)
    HW_PF_MID, // count read; 4 writes, H1, 3 writes, K, use
    HW_AFTER,  // count read; 7 writes, H1, K, use
    HW_PF_DS,  // count read; DS2, H1 through it, 7 writes, K, use (a forwarding channel's next header)
    HW_PF_SEQ, // count read; H1 used at once, then 7 writes, K (no prefetch: the header read at the next send's start)
    // The post-send group (TX-queue status: command read discarded, status; the completion counter, an L1 word) with
    // and without the next header in it. Cost against HP_BASE.
    HP_BASE,   // count read; K, 3 uses of filler
    HP_GRP,    // count read; T Q + completion counter, K, use status and counter
    HP_GRP_PF, // count read; T Q + completion counter + H1, K, use all three
    // Which load form costs what after the count read (HB_H1's lone lhu cost 2, RESULTS §17's lone L1 lw 7-8). Cost
    // against HB_BASE. L = RESULTS §17's L1 word.
    HX_LW40,     // count read; lw of H0's word 40, K, use
    HX_LHU_L,    // count read; lhu of L, K, use
    HX_LW_L,     // count read; lw of L, K, use
    HX_LHU40_AT, // count read; lhu of H0's word 40 used at once, K
    HX_LW40_AT,  // count read; lw of H0's word 40 used at once, K
    HX_LW_L_AT,  // count read; lw of L used at once, K
    HX_LW_L_NC,  // no count read; lw of L, K, use (RESULTS §17's "L, K, use" in this block shape; against HX_BASE_NC)
    HX_LHU40_NC, // no count read; lhu of H0's word 40, K, use (against HX_BASE_NC)
    HX_BASE_NC,  // no count read; K, 4 uses of filler
    // The send's pre-send group as the router has it (TX-queue status: command read discarded, then status; and the
    // packet's header word), used at once, against the status alone: what the header costs inside that group.
    // Against HB_BASE.
    HS_TQ_AT,    // count read; T Q, status used at once, K
    HS_TQH_AT,   // count read; H0 T Q, header and status used at once, K
    HS_TQH2_AT,  // count read; H0 T Q used at once, then H1 T Q used at once (two sends' pre-send groups), K
    kNumPatterns,
};

constexpr uint32_t kFill[] = {0, 2, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48};
constexpr size_t kNumFill = sizeof(kFill) / sizeof(kFill[0]);
constexpr uint32_t kIterations = 1024;
constexpr uint32_t kRepeats = 3;  // keep the fastest run
constexpr uint32_t kStream = 0;   // stream registers read (fabric is off, so nothing else uses them)
constexpr uint32_t kStream2 = 1;
constexpr uint32_t kStream3 = 2;
constexpr uint32_t kStream4 = 3;
// The patterns do not all fit the 24 KB kernel code limit at once; the host runs the kernel once per group,
// selected by ERISC_LOAD_GROUP (patterns [first, last) of each group).
constexpr uint32_t kGroups[][2] = {
    {0, PRE_BOTH_REV},
    {PRE_BOTH_REV, BASE3},
    {BASE3, BASE5},
    {BASE5, DEP_CMDBUF},
    {DEP_CMDBUF, BASE7},
    {BASE7, HB_BASE},
    {HB_BASE, HB_H2_SEQ},
    {HB_H2_SEQ, HW_BASE},
    {HW_BASE, HP_BASE},
    {HP_BASE, HX_LW40},
    {HX_LW40, HX_LW40_AT},
    {HX_LW40_AT, HS_TQ_AT},
    {HS_TQ_AT, kNumPatterns}};
constexpr uint32_t kNumGroups = 13;
constexpr uint32_t kTxq = 1;     // TX queue whose registers are read (the receiving ERISC's ack queue)
constexpr uint32_t kNiuNoc = 1;  // NoC whose status registers are read (the router's local-write NoC)

constexpr uint32_t kResultsOffset = 0x0;  // from the unreserved base
constexpr uint32_t kL1WordOffset = 0x1000;
constexpr uint32_t kL1Word2Offset = 0x1040;
// The send-batch patterns' four ring slots (4448 B apart: a 4352 B packet's slot) and the header word they read.
constexpr uint32_t kSlotsOffset = 0x2000;
constexpr uint32_t kSlotStride = 4448;
constexpr uint32_t kHeaderWordOffset = 40;
// Results: [pattern][fill] cycles for kIterations blocks, then acc, done flag, CSR 0x7c0 at kernel start.
constexpr uint32_t kResultWords = kNumPatterns * kNumFill + 3;
constexpr uint32_t kDone = 0xD0D0;

}  // namespace erisc_load_overlap
