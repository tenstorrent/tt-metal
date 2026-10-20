// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Layout shared by the DRISC microbenchmark host test (test_drisc_microbench.cpp) and its two kernels: one on a
// DRISC, one on an ERISC of an active Ethernet core. All offsets are from each core's L1 UNRESERVED base.
//
// Each result entry is a pair of words {cycles, ops}; the host divides. Cycles are the kernel's timer: mcycle on both
// cores, or the RISC-V debug wall clock on the DRISC when built with DRISC_MB_WALL_CLOCK.

#pragma once

#include <cstdint>

namespace drisc_mb {

constexpr uint32_t kMagicReady = 0xD15C0001u;

// L1 layout (same offsets on both cores).
constexpr uint32_t kReadyOffset = 0;          // the peer writes kMagicReady here once its streams are reset
constexpr uint32_t kHandshakeSrcOffset = 16;  // source word for writing kMagicReady to the peer
constexpr uint32_t kScratchOffset = 64;       // 64 B of scratch for load chains
constexpr uint32_t kResultsOffset = 128;      // uint32_t[2 * kMaxResults]
constexpr uint32_t kMaxResults = 112;
constexpr uint32_t kBufferOffset = 1024;  // packet buffers
constexpr uint32_t kSlotBytes = 4096;     // payload slot size; payloads up to this size

// DRISC: staging ring of `num_slots` slots at kBufferOffset (slot 0 is also the source of DRISC writes).
// ERISC: one source buffer at kBufferOffset, then a landing ring of kEthLandingSlots slots.
constexpr uint32_t kEthSrcOffset = kBufferOffset;
constexpr uint32_t kEthLandingOffset = kBufferOffset + kSlotBytes;
constexpr uint32_t kEthLandingSlots = 16;
constexpr uint32_t kEthL1Bytes = kEthLandingOffset + kEthLandingSlots * kSlotBytes;
constexpr uint32_t kMaxDriscSlots = 16;
constexpr uint32_t drisc_l1_bytes(uint32_t num_slots) { return kBufferOffset + num_slots * kSlotBytes; }

// Stream registers (a DRAM tile has 16, an Ethernet tile 32). Counters only ever count up within a phase and are
// reset between phases, so no update races with a reset.
constexpr uint32_t kDriscSyncStream = 12;      // phase barrier: the ERISC increments it
constexpr uint32_t kDriscDoorbellStream = 13;  // packets the ERISC has pushed / pings
constexpr uint32_t kDriscScratchStream = 14;   // local read tests; target of the ERISC's bare doorbell test
constexpr uint32_t kDriscScratchStream2 = 15;
constexpr uint32_t kEthSyncStream = 24;      // phase barrier: the DRISC increments it
constexpr uint32_t kEthDoorbellStream = 25;  // copies the DRISC has landed / pings
constexpr uint32_t kEthCreditStream = 26;    // staging slots the DRISC has freed
constexpr uint32_t kEthScratchStream = 27;   // local read tests; target of the DRISC's bare doorbell test

constexpr uint32_t kMaxDepths = 6;         // staging depths swept by the pipeline phase
constexpr uint32_t kPipeModes = 2;         // pipeline issue: 0 stateless, 1 lean stateful (lean_trid_write etc.)
constexpr uint32_t kSpinLimit = 1u << 27;  // polls before a wait gives up (~1 s); the core then reports an error

// Runtime args, both kernels (the peer is the other core).
enum Arg : uint32_t {
    ARG_MY_BASE,
    ARG_PEER_X,
    ARG_PEER_Y,
    ARG_PEER_BASE,
    ARG_PAYLOAD,      // bytes per packet (<= kSlotBytes)
    ARG_N,            // back-to-back operations per throughput entry
    ARG_N_RT,         // round trips per latency entry
    ARG_DRISC_NOC1,   // 1: the DRISC's NIU1 is in stream mode, so NoC1 tests run (DRISC writes, ERISC pushes)
    ARG_DRISC_SLOTS,  // DRISC staging slots that fit in its L1 (<= kMaxDriscSlots)
    ARG_PIPE_PKTS,    // packets per pipeline run
    ARG_FWD_ERISCS,   // other Ethernet cores the DRISC forwards into in part 3 (0-2; they run no kernel)
    ARG_FWD0_X,       // their virtual coordinates
    ARG_FWD0_Y,
    ARG_FWD1_X,
    ARG_FWD1_Y,
    ARG_ROLE,      // part 4: 0 pulls from the target ERISC, 1 writes into it
    ARG_TARGET_X,  // part 4: the ERISC both DRISCs work on (the peer is then the other DRISC)
    ARG_TARGET_Y,
    ARG_TARGET_BASE,
    ARG_TWO_MASK,  // part 4: bit v runs variant v of kTwoDriscList
    ARG_NUM_DEPTHS,
    ARG_DEPTHS,  // kMaxDepths staging depths follow
    ARG_COUNT = ARG_DEPTHS + kMaxDepths,
};

// DRISC results.
enum DriscResult : uint32_t {
    D_ERROR,                 // {phase that timed out + 1, 0}; 0 = ok
    D_SINK,                  // keeps measured loads alive
    D_TIMER_READ,            // the kernel's timer read
    D_WALL_CLOCK_READ,       // RISC-V debug wall clock read (get_timestamp_32b)
    D_CALIBRATION,           // {timer delta, wall-clock delta} over the same interval
    D_L1_LOAD_DEP,           // dependent L1 load (pointer chase)
    D_L1_LOAD_INDEP4,        // four independent L1 loads issued back to back, then used (per load)
    D_L1_POLL_FENCE,         // fence + L1 load: the credit-counter poll of today's DRISC kernels
    D_STREAM_READ,           // own stream register read: a doorbell poll
    D_STREAM_READ_INDEP4,    // four independent stream register reads, then used (per read)
    D_STREAM_LOCAL_UPDATE,   // local stream register update (consume a doorbell)
    D_CMD_BUF_READY,         // NoC command buffer ready read
    D_CHECK_PUSH_DOORBELLS,  // {doorbells seen, expected} after the ERISC's push phase
    D_CHECK_BARE_DOORBELLS,  // {bare doorbells seen, expected}
    D_NOC_BASE,              // per-NoC block (DriscNocResult), NoC0 then NoC1
};
enum DriscNocResult : uint32_t {
    DN_WR64_ISSUE,  // 64 B non-posted write into ERISC L1, back to back: issue cycles per write
    DN_WR64_DRAIN,  //   ... including the wait for all acks
    DN_WR_ISSUE,    // payload-size non-posted write
    DN_WR_DRAIN,
    DN_WR_POSTED_ISSUE,  // payload-size posted write
    DN_WR_POSTED_DRAIN,  //   ... until all have left the NIU
    DN_DOORBELL_ISSUE,   // inline write to an ERISC stream register (a router doorbell)
    DN_DOORBELL_DRAIN,
    DN_FWD_ISSUE,  // a dumb forwarder's copy: payload write with a transaction id + doorbell
    DN_FWD_DRAIN,
    DN_SF_FWD_ISSUE,    // the same copy as a dedicated forwarder would issue it: stateful write (destination set
    DN_SF_FWD_DRAIN,    //   once) with a transaction id + doorbell, back to back
    DN_SF_FWD_ALONE,    // one stateful copy at a time with the NoC idle: CPU cycles when the port is free
    DN_DOORBELL_ALONE,  // one doorbell at a time with the NoC idle
    DN_LEAN_FWD_ISSUE,  // the copy with lean stateful issue (lean_trid_write + lean_doorbell), back to back
    DN_LEAN_FWD_DRAIN,
    DN_LEAN_FWD_ALONE,  //   ... one at a time with the NoC idle
    DN_WR_RTT,          // one payload write, then wait for its ack
    DN_RD_ISSUE,        // payload-size read from ERISC L1 (a pull), back to back
    DN_RD_DRAIN,
    DN_RD_RTT,  // one read, then wait for the data
    DN_COUNT,
};
constexpr uint32_t D_PIPE_BASE = D_NOC_BASE + 2 * DN_COUNT;
enum DriscPipeResult : uint32_t {
    DP_BUSY,        // cycles in iterations with a packet or credit pending, per packet
    DP_WORK,        //   ... of which iterations that forwarded or credited something (the rest only waited
                    //   for writes to leave L1)
    DP_IDLE_POLLS,  // doorbell polls that found nothing to do, per packet
    DP_COUNT,
};
// Pipeline results: mode-major, then depth.
constexpr uint32_t D_COUNT = D_PIPE_BASE + kPipeModes * kMaxDepths * DP_COUNT;
static_assert(D_COUNT <= kMaxResults);

// Part 3 (DRISC command buffers and NoC ports) runs as its own program and reuses result indices from D_NOC_BASE;
// the host keeps its results apart. Payload writes go into the ERISC's landing ring, reads come from its source buffer.
// Each NIU has 4 command buffers, all the DRISC's own. Writes use at most 3: the read buffer keeps the read control
// word firmware sets at boot (the read path never rewrites it), and a pulling DRISC needs it for reads.
constexpr uint32_t kMaxWriteCmdBufs = 3;
enum DriscCmdBufNocResult : uint32_t {  // per NoC
    DCN_HOLD,                           // one write with the NoC idle: cycles until its command buffer is free
    DCN_WR_K,                           // back-to-back writes round-robin over k = 1..3 command buffers:
                                        //   {issue, drain} pairs, k-major
    DCN_COUNT = DCN_WR_K + 2 * kMaxWriteCmdBufs,
};
constexpr uint32_t DC_BOTH = D_NOC_BASE + 2 * DCN_COUNT;  // writes alternating NoC0/NoC1, k = 1, 2 per NoC: pairs
constexpr uint32_t kDuplexVariants = 4;  // reads + writes interleaved: {read NoC, write NoC, write command bufs}
constexpr uint32_t DC_DUPLEX = DC_BOTH + 2 * 2;  // per variant: {issue, drain}
// Forwarding into other Ethernet cores (B, C) while pulling from the benchmark's ERISC (A), as a router forwards to
// its downstream routers. Variants, per packet: {pull from A on NoC0?, copy to B on NoC b (-1: none), copy to C on NoC
// c (-1: none)}; B and C need ARG_FWD_ERISCS >= 1 / 2, NoC1 needs the DRISC's NIU1 in stream mode.
struct FwdVariant {
    bool pull;
    int8_t b_noc;
    int8_t c_noc;
};
constexpr uint32_t kFwdVariants = 5;
constexpr FwdVariant kFwdVariantList[kFwdVariants] = {
    {false, 0, -1},  // copies to B only (reference)
    {true, 0, -1},   // pull + one copy, same NoC
    {true, 1, -1},   // pull + one copy, other NoC
    {true, 0, 1},    // pull + two copies (multicast), one per NoC
    {true, 1, 1},    // pull + two copies, both on the other NoC
};
constexpr uint32_t DC_FWD = DC_DUPLEX + 2 * kDuplexVariants;  // per variant: {issue, drain}
constexpr uint32_t DC_COUNT = DC_FWD + 2 * kFwdVariants;
static_assert(DC_COUNT <= kMaxResults);

// Part 4 (two DRISCs on one Ethernet tile, as when its link carries traffic both ways): DRISC role 0 pulls 4 KiB
// packets from the ERISC's source buffer (its received packets), role 1 writes 4 KiB packets into its landing ring
// (its packets to send). Each DRISC puts {cycles, packets} for its own side at D_NOC_BASE + variant.
struct TwoDriscVariant {
    int8_t pull_noc;       // -1: the pulling DRISC sits this one out
    int8_t write_noc;      // -1: the writing DRISC sits this one out
    int8_t own_write_noc;  // >= 0: the pulling DRISC also writes, interleaved with its pulls (one DRISC, both ways)
};
constexpr uint32_t kTwoDriscVariants = 7;
constexpr TwoDriscVariant kTwoDriscList[kTwoDriscVariants] = {
    {0, -1, -1},  // pull alone
    {-1, 0, -1},  // write alone
    {0, 0, -1},   // both DRISCs at once, same NoC
    {0, 1, -1},   // both at once, pull on NoC0, write on NoC1
    {1, 0, -1},   // both at once, pull on NoC1, write on NoC0
    {0, -1, 0},   // DRISC 1 pulls and writes, NoC0
    {0, -1, 1},   // DRISC 1 pulls on NoC0, writes on NoC1
};
constexpr uint32_t kTwoDriscOpsScale = 4;  // 4 * ARG_N packets each, so the two DRISCs' start skew is negligible

// ERISC results.
enum EthResult : uint32_t {
    E_ERROR,
    E_SINK,
    E_TIMER_READ,
    E_L1_LOAD_DEP,
    E_L1_LOAD_INDEP4,
    E_STREAM_READ,
    E_STREAM_READ_INDEP4,  // the read-overlap question for polling several channels at once
    E_WR_ISSUE,            // payload write into DRISC L1 (a push), back to back
    E_WR_DRAIN,
    E_DOORBELL_ISSUE,  // inline write to a DRISC stream register
    E_DOORBELL_DRAIN,
    E_PUSH_ISSUE,  // push = payload write + doorbell, as the receiving ERISC would hand a packet over
    E_PUSH_DRAIN,
    E_SF_PUSH_ISSUE,    // push as a receiving ERISC with a dedicated DRISC would issue it: stateful write
    E_SF_PUSH_DRAIN,    //   (destination set once) with a transaction id + doorbell, back to back
    E_SF_PUSH_ALONE,    // one at a time with the NoC idle (the wait is not timed): CPU cycles when the port is free
    E_SF_WR_ALONE,      //   ... the stateful write alone
    E_WR_ALONE,         //   ... a stateless write alone
    E_DOORBELL_ALONE,   //   ... a doorbell alone
    E_PUSH_ALONE,       //   ... a stateless push (write + doorbell) alone
    E_LEAN_PUSH_ISSUE,  // push with lean stateful issue (lean_trid_write + lean_doorbell), back to back
    E_LEAN_PUSH_DRAIN,
    E_LEAN_PUSH_ALONE,      //   ... one at a time with the NoC idle
    E_LEAN_WR_ALONE,        //   ... the lean write alone
    E_LEAN_DOORBELL_ALONE,  //   ... the lean doorbell alone
    E_WR_RTT,               // one payload write into DRISC L1, then wait for its ack
    E_PING_RTT,             // ERISC -> DRISC doorbell, DRISC -> ERISC doorbell
    E_PING_DATA_RTT,        // the same with a payload each way
    E_CHECK_FWD_DOORBELLS,  // {doorbells seen, expected} after the DRISC's write phase
    E_CHECK_BARE_DOORBELLS,
    E_PIPE_BASE,
};
enum EthPipeResult : uint32_t {
    EP_TOTAL,         // cycles from the first push until every copy has landed back, per packet
    EP_BUSY,          // cycles spent pushing, per packet
    EP_CREDIT_WAITS,  // credit polls that found the staging ring full, per packet
    EP_COUNT,
};
constexpr uint32_t E_COUNT = E_PIPE_BASE + kPipeModes * kMaxDepths * EP_COUNT;
static_assert(E_COUNT <= kMaxResults);

}  // namespace drisc_mb
