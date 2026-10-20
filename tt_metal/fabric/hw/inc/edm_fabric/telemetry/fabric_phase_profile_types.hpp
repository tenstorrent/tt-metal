// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Per-ERISC phase accounting for fabric router throughput analysis (experimental).
//
// Host side: TT_FABRIC_PHASE_PROFILE=<level> enables it (1 = counters plus per-iteration busy/blocked/idle
// accounting, 2 = level 1 plus phase timers, 3 = level 2 plus the primitive-cost microbenchmark).
// TT_FABRIC_PHASE_MASK selects which phases are timed at level >= 2 (bit per FabricPhase, default all); phase
// occurrence counts are kept for every phase at level >= 1. TT_FABRIC_PHASE_NO_ITER=1 drops the per-iteration
// timestamp, leaving counters only; that still slows a 150-190 cycle/packet router by 15-30%.
// TT_FABRIC_TXQ_COUNTERS=1 samples the Ethernet TX queue registers (FabricTxqRegister); on its own (level 0) it
// adds nothing to the hot path. Each ERISC owns one FabricPhaseProfile block at
// PHASE_PROFILE_BUFFER_ADDR + erisc_id * sizeof(FabricPhaseProfile). The router writes running totals, so
// the host reads the block before and after a test and subtracts.
// With EXP_DRISC_OFFLOAD, TT_FABRIC_PHASE_PROFILE also profiles each router's DRISC (PHASE_PROFILE_DRISC_BLOCK).
// TT_FABRIC_DRISC_PHASE_PROFILE=<level> sets the DRISC's level on its own: alone, it profiles the DRISC while the
// ERISCs run unprofiled, so the DRISC is measured at full traffic rate. Level 3 on the DRISC is level 2.
//
// Experiment knobs (all default off), passed to the router as compile-time args:
//   TT_FABRIC_EXP_FLAGS    bitmask of FabricExperimentFlags
//   TT_FABRIC_EXP_TX_DELAY extra core cycles spun per packet sent on the TX path
//   TT_FABRIC_EXP_RX_DELAY extra core cycles spun per packet consumed on the RX path
//   TT_FABRIC_EXP_DRISC_DELAY extra core cycles the offload DRISC spins per packet it handles
//
// All durations are ERISC core clock cycles (mcycle).

enum FabricPhase : uint32_t {
    // TX (sender channel) phases
    PHASE_TX_SEND = 0,        // can_send decision taken -> remote pointer update issued
    PHASE_TX_TXQ_SPIN = 1,    // spin on eth txq between payload send and pointer update
    PHASE_TX_COMPLETION = 2,  // completion/ack processing that returned at least one credit
    PHASE_TX_CONN = 3,        // connection liveness / teardown checks that ran
    // RX (receiver channel) phases
    PHASE_RX_DECODE_ADMIT = 4,  // packet seen -> admission decided (decode, downstream credit, trid check)
    PHASE_RX_FORWARD = 5,       // NoC writes to downstream routers
    PHASE_RX_LOCAL = 6,         // local-chip delivery
    PHASE_RX_COMPLETION = 7,    // flush check + completion ack that sent an ack
    PHASE_RX_ACK = 8,           // first-level ack that sent an ack
    PHASE_TX_TXQ_PRESPIN = 9,   // spin on eth txq before the payload send (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA)
    // Whole channel visits, including the phases above that run inside them
    PHASE_TX_VISIT = 10,  // one serviced sender channel step (cache invalidate through connection checks)
    PHASE_RX_VISIT = 11,  // one serviced receiver channel step (packet poll through completion ack)
    PHASE_COUNT = 12,
};
constexpr uint32_t PHASE_MASK_ALL = (1u << PHASE_COUNT) - 1;

enum FabricPhaseEvent : uint32_t {
    EVENT_TX_BLOCKED_CREDIT = 0,  // sender channel had a packet but the remote receiver had no free slot
    EVENT_TX_BLOCKED_TXQ = 1,     // sender channel had a packet and credit but the eth txq was busy
    EVENT_RX_BLOCKED_ADMIT = 2,   // receiver had a packet but a downstream/local output was full
    EVENT_RX_BLOCKED_TRID = 3,    // receiver had a packet but its slot's previous NoC writes were not sent
    EVENT_COUNT = 4,
};

// Ethernet TX queue registers sampled by the router with TT_FABRIC_TXQ_COUNTERS=1 (at any profile level, including
// 0, where nothing else is recorded and the router is not perturbed). Each ERISC samples only the queues it issues
// commands on. The first TXQ_COUNTER_COUNT values are free-running 32-bit hardware counters (the host subtracts
// modulo 2^32); the rest are configuration registers, reported as read.
enum FabricTxqRegister : uint32_t {
    TXQ_TRANSFERS = 0,     // ETH_TXQ_TRANSFER_CNT: commands issued
    TXQ_PKTS_STARTED = 1,  // ETH_TXQ_PKT_START_CNT: Ethernet packets issued
    TXQ_PKTS_SENT = 2,     // ETH_TXQ_PKT_END_CNT: Ethernet packets sent
    TXQ_WORDS = 3,         // ETH_TXQ_WORD_CNT: 16-byte words sent
    TXQ_RESENDS = 4,       // ETH_TXQ_RESEND_CNT: resend start events
    TXQ_COUNTER_COUNT = 5,
    TXQ_CTRL = 5,                      // ETH_TXQ_CTRL
    TXQ_MAX_PKT_SIZE = 6,              // ETH_TXQ_MAX_PKT_SIZE_BYTES: largest Ethernet payload
    TXQ_BURST_LEN = 7,                 // ETH_TXQ_BURST_LEN
    TXQ_REMOTE_SEQ_TIMEOUT = 8,        // ETH_TXQ_REMOTE_SEQ_TIMEOUT: resend timeout without sequence acks
    TXQ_LOCAL_SEQ_UPDATE_TIMEOUT = 9,  // ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT: sequence update when otherwise idle
    TXQ_MIN_PKT_WORDS = 10,            // ETH_TXQ_MIN_PACKET_SIZE_WORDS
    TXQ_ACCEPT_AHEAD = 11,             // ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD
    TXQ_REGISTER_COUNT = 12,
};
constexpr uint32_t PHASE_PROFILE_TXQS = 3;  // Blackhole has three Ethernet TX queues (Wormhole two)
// TX queue sampling in the otherwise disabled profiler (level 0) runs on every this-many outer loop iterations.
constexpr uint32_t PHASE_PROFILE_TXQ_SAMPLE_PERIOD = 64;

enum FabricExperimentFlags : uint32_t {
    EXP_NONE = 0,
    // Drop local-chip delivery of pure data writes (plain writes, all-write scatters); atomics and inline writes
    // are still delivered so semaphores and test sync work. The slot still retires. Upper bound on what any
    // local-delivery offload can buy. Breaks data delivery: benchmark mode only.
    EXP_RX_SKIP_LOCAL_DELIVERY = 1u << 0,
    // Deliver NOC_UNICAST_WRITE packets with one bare stateless write (no send-type dispatch); other send types
    // take the normal path. Isolates the dispatch overhead around the NoC write. Benchmark mode only.
    EXP_RX_LOCAL_AS_SINGLE_WRITE = 1u << 1,
    // Return credits to upstream senders (workers and routers) in batches: when a channel has accumulated
    // min(4, slots/2) credits, or on any visit that does not send. Approximates moving credit return off the ERISC.
    EXP_TX_AMORTIZE_WORKER_CREDITS = 1u << 2,
    // Poll connection open/teardown state on every 16th visit of a sender channel instead of every visit.
    EXP_TX_LAZY_CONN_CHECK = 1u << 3,
    // Replace local delivery of pure data writes with one stateful NoC write of the whole packet (header +
    // payload) into a scratch region of this core's L1: the ERISC side of a push handoff to a helper core, minus
    // the doorbell. Benchmark mode only.
    EXP_RX_LOCAL_AS_PUSH = 1u << 4,
    // Speedy TX: skip the eth-txq busy read before the payload send whose result is unused when
    // ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA is off (the read is volatile, so the compiler keeps it).
    EXP_TX_DROP_DEAD_TXQ_READ = 1u << 5,
    // Speedy and generic TX: skip the eth-txq busy check in the can-send decision. Unsafe in general (the next
    // command could be written while the queue is busy); measures the cost of that read. Benchmark mode only.
    EXP_TX_NO_SEND_TXQ_CHECK = 1u << 6,
    // Generic TX: read a sender channel's completion (and first-level ack) counter only while the channel has
    // packets the receiver has not completed. A completion can only follow a send, so this skips no event.
    EXP_TX_LEAN_COMPLETIONS = 1u << 7,
    // Generic TX: read the free-slot stream registers of up to four VC0 sender channels back to back at the top of
    // each loop iteration, so their latencies overlap, and use the values in the channel visits (RESULTS §17).
    EXP_TX_GROUPED_PACKET_CHECK = 1u << 8,
    // Two-ERISC generic path: E1 handles the upstream side of the sender channels E0 services (completions turned
    // into credits for the upstream worker or router, and worker connection open/teardown). E0 still reads the
    // completion counters for its own receiver-slot accounting.
    EXP_TX_UPSTREAM_ON_E1 = 1u << 9,
    // Generic TX send (send_next_data). Skip the worker interface's write counter update: the router never reads it.
    EXP_TX_SEND_NO_WRITE_COUNTER = 1u << 10,
    // Generic TX send: issue the TX-queue busy read before the slot bookkeeping and test it before the pointer update,
    // as the speedy path does, instead of reading it after the bookkeeping and using it at once.
    EXP_TX_SEND_EARLY_TXQ_READ = 1u << 11,
    // Generic TX send: advance the cached slot address by the slot stride (wrapping at the last slot) instead of an
    // index wrap plus a lookup in the slot address table.
    EXP_TX_SEND_STRIDE_SLOT = 1u << 12,
    // Generic TX send: mark the sent packet's slot in the free-slot stream register through the channel's
    // compile-time stream id (a constant address) instead of the run-time one.
    EXP_TX_SEND_CT_FREE_SLOT_STREAM = 1u << 13,
    // Generic TX: read the next packet's size from its header before the TX-queue can-send check, so the L1 read
    // overlaps the TX-queue register reads, instead of inside the send.
    EXP_TX_SEND_EARLY_SIZE_READ = 1u << 14,
    // Generic RX (receiving ERISC, two-ERISC two-TX-queue mode): at the end of each receiver step, issue its
    // completion reads (the oldest uncompleted slot's write-sent status, the TX queue's busy check) together with the
    // next step's packets-received count, so their latencies overlap.
    EXP_RX_GROUPED_READS = 1u << 15,
    // With EXP_RX_GROUPED_READS: keep reading the packets-received count at the start of the step instead of taking
    // it from the previous step's group (the group then has the completion reads only).
    EXP_RX_GROUPED_NO_PREFETCH = 1u << 16,
    // With EXP_RX_LOCAL_AS_PUSH: issue the push with one command-buffer poll (lean stateful issue, length written per
    // packet) and follow it with a doorbell: a stateless register write to an unused stream register on this tile.
    // A real design would give the doorbell its own command buffer (~8 cycles cheaper); here none is free.
    EXP_RX_PUSH_LEAN = 1u << 17,
    // Generic TX, with EXP_TX_LEAN_COMPLETIONS and EXP_TX_GROUPED_PACKET_CHECK: skip the visits to the forwarding
    // channels in the grouped read (VC0 channels 1-3) in an iteration where none has a packet in the grouped read and
    // none has packets awaiting completion (one in-flight count for the set). One branch for all of them. Only on
    // routers where those channels skip the connection liveness check (router-to-router channels).
    EXP_TX_SKIP_IDLE_CHANNELS = 1u << 18,
    // Same conditions, per channel: a forwarding channel in the grouped read with no packet and no packets awaiting
    // completion returns at the top of its visit (one test of the grouped value, one of its own sent count).
    EXP_TX_SKIP_IDLE_PER_CHANNEL = 1u << 19,
    // Push to this router's own DRISC instead of this tile's L1 (the builder assigns each
    //  router on a chip a different Metal DRISC whose NIU on the local-write NoC is in stream mode). Payloads go to a
    //  ring of slots in the DRISC's unreserved L1; each push is followed by a stateful doorbell (+1 on a DRISC stream
    //  register) on read_cmd_buf in 2D (write_at_cmd_buf in 1D). With EXP_RX_LOCAL_AS_PUSH, replaces local delivery
    //  only (lean with EXP_RX_PUSH_LEAN, stock otherwise). No DRISC kernel runs, so no slot credits come back.
    //  Benchmark mode only.
    EXP_RX_PUSH_DRISC = 1u << 20,
    // Offload E1's receiver decode, admission, downstream router forwarding (on NoC0), and local delivery (on NoC1)
    // to a paired dual-NoC Metal DRISC core (the P5 receive pipeline). E1 pushes arriving Ethernet slots into a
    // 16-slot staging ring in DRISC L1, DRISC connects directly to downstream E0s for credit exchange, and the
    // offloaded decode/forward/local-delivery code is compiled out of E1.
    EXP_DRISC_OFFLOAD = 1u << 21,
    EXP_RX_BLIND_PUSH_DRISC = EXP_DRISC_OFFLOAD,
    // Generic RX (fused flush/completion, no first-level ack): give receiver channel 0 all 16 hardware transaction IDs
    // (1 per slot in a 16-slot ring, so trid == slot_index with no wrap collision), skip the trid-flushed and TXQ1
    // busy reads when nothing is ready to retire, and batch completion acks by up to 4 slots (checked once a 5th slot
    // has been issued so the first 4 have already left cmd_buf 0, or immediately when no new packet arrived).
    EXP_RX_BATCH_COMPLETION = 1u << 22,
    // Generic TX send: issue the post-payload TX-queue busy check (CMD + STATUS reads) together with the sender
    // channel's L1 completion-counter read after slot-pointer bookkeeping, followed by 6 ALU instructions (the K = 5..7
    // overlap window from RESULTS §17) before testing both, avoiding the separate L1 completion read and stack reloads
    // after the send.
    EXP_TX_SEND_OVERLAP_COMPLETION = 1u << 23,
    // Generic TX VC0 loop: compile-time slot bounds/stride (with LeanStrideSenderEthChannel dropping the dead
    // SlotAddresses stack table and runtime bound fields when stride slot advancing is enabled), compile-time remote
    // receiver slot bounds, and compile-time free-slot stream address in the overlapped send tail.
    EXP_TX_LEAN_VC0_LOOP = 1u << 24,
    // Generic TX per-channel idle check (with EXP_TX_SKIP_IDLE_PER_CHANNEL): mark the idle early-return [[likely]] so
    // the compiler keeps the idle path inline instead of jumping out of line.
    EXP_TX_IDLE_LIKELY = 1u << 25,
};

struct FabricPhaseProfile {
    uint64_t wall_ts;        // eth wall clock at last flush
    uint64_t cycle_ts;       // core cycle counter at last flush (with wall_ts, calibrates the two clocks)
    uint64_t iters;          // inner main-loop iterations
    uint64_t busy_ticks;     // iterations that made progress
    uint64_t blocked_ticks;  // iterations with pending work and no progress
    uint64_t idle_ticks;     // idle stretches shorter than PHASE_PROFILE_IDLE_GAP_TICKS
    uint64_t gap_ticks;      // idle stretches at least PHASE_PROFILE_IDLE_GAP_TICKS long (between tests)
    uint64_t pkts;           // packets sent (TX) or consumed (RX)
    // Static info: bits 0-7 serviced sender channels, 8-15 serviced receiver channels, 16-31 phase mask,
    // 32-39 profile mode.
    uint64_t info;
    // Timer self-cost from an in-situ calibration at start-up, x100: bits 0-31 cycles one empty open/close pair adds
    // to the phase it measures, bits 32-63 cycles it adds to everything enclosing it.
    uint64_t timer_cal;
    uint64_t events[EVENT_COUNT];
    uint64_t phase_ticks[PHASE_COUNT];
    uint64_t phase_count[PHASE_COUNT];
    // Bit q set: txq[q] holds samples of Ethernet TX queue q (TT_FABRIC_TXQ_COUNTERS).
    uint64_t txq_mask;
    // TT_FABRIC_EXP_STACK_PAINT: the stack pointer inside the router's kernel_main (its frame allocated), the lowest
    // stack word written so far, and the limit the stack can grow down to (end of the kernel's local-data-memory
    // sections). stack_painted is 1 once the router has filled its free stack.
    uint32_t stack_main_sp;
    uint32_t stack_lowest_used;
    uint32_t stack_limit;
    uint32_t stack_painted;
    uint64_t reserved;  // keeps the size a multiple of 16
    uint64_t txq[PHASE_PROFILE_TXQS][TXQ_REGISTER_COUNT];
};

static_assert(sizeof(FabricPhaseProfile) % 16 == 0);

// Primitive-cost microbenchmark results (TT_FABRIC_PHASE_PROFILE=3): average core cycles x100 per operation,
// one uint32 per entry at PHASE_PROFILE_BUFFER_SIZE past the profile buffer, written by ERISC 1 at router start.
enum FabricMicrobench : uint32_t {
    MB_CYCLE_CSR_READ = 0,
    MB_L1_LOAD_DEPENDENT = 1,
    MB_STREAM_REG_READ = 2,
    MB_ETH_TXQ_BUSY_READ = 3,
    MB_WALL_CLOCK_READ = 4,
    MB_NOC_CMD_BUF_READY_READ = 5,
    MB_NOC_WRITE_64B = 6,
    MB_NOC_WRITE_2KB = 7,
    MB_NOC_WRITE_STATEFUL_64B = 8,
    MB_NOC_INLINE_L1_WRITE = 9,
    // Straight-line code of N instructions executed repeatedly from L1: cycles per instruction (x100). A jump once
    // N exceeds the L0 instruction cache shows the cache size and the cost of fetching code from L1.
    MB_ICACHE_64 = 10,
    MB_ICACHE_128 = 11,
    MB_ICACHE_256 = 12,
    MB_ICACHE_512 = 13,
    MB_ICACHE_1024 = 14,
    MB_ICACHE_2048 = 15,
    MB_SINK = 16,
    MB_DONE = 17,
    MB_COUNT = 18,
};
constexpr uint32_t PHASE_PROFILE_MICROBENCH_RESULTS_SIZE = 128;
// Scratch for the microbenchmark's L1 loads and loopback NoC writes (sources at +0, destinations at +4096).
constexpr uint32_t PHASE_PROFILE_MICROBENCH_SCRATCH_SIZE = 8192;
constexpr uint32_t PHASE_PROFILE_MICROBENCH_SIZE =
    PHASE_PROFILE_MICROBENCH_RESULTS_SIZE + PHASE_PROFILE_MICROBENCH_SCRATCH_SIZE;
static_assert(MB_COUNT * sizeof(uint32_t) <= PHASE_PROFILE_MICROBENCH_RESULTS_SIZE);

// Idle stretches at least this long are treated as gaps between tests rather than idle time inside one.
constexpr uint32_t PHASE_PROFILE_IDLE_GAP_TICKS = 20000;
// Counters-only mode (no per-iteration timestamps) detects gaps by consecutive idle iterations instead.
constexpr uint32_t PHASE_PROFILE_IDLE_GAP_ITERS = 256;
constexpr uint32_t PHASE_PROFILE_MAX_ERISCS = 2;
// With EXP_DRISC_OFFLOAD, block PHASE_PROFILE_DRISC_BLOCK holds the profile of the router's DRISC. The DRISC stages it
// in its own L1 and copies it into E1's buffer every PHASE_PROFILE_DRISC_FLUSH_ITERS loop iterations, so the host reads
// it with the ERISC blocks and never reads DRAM-tile L1. Durations are DRISC core cycles (same clock as the ERISCs).
// The RX phases and events keep their meaning: decode/admit, forward (NoC0), local delivery (NoC1), and
// rx_completion = ring credits returned to E1; rx_blk_trid = a credit batch waiting on its slots' NoC writes.
constexpr uint32_t PHASE_PROFILE_DRISC_BLOCK = PHASE_PROFILE_MAX_ERISCS;
constexpr uint32_t PHASE_PROFILE_BLOCKS = PHASE_PROFILE_MAX_ERISCS + 1;
constexpr uint32_t PHASE_PROFILE_DRISC_FLUSH_ITERS = 256;
constexpr uint32_t PHASE_PROFILE_BUFFER_SIZE = PHASE_PROFILE_BLOCKS * sizeof(FabricPhaseProfile);

// Scratch for EXP_RX_LOCAL_AS_PUSH's loopback write (without EXP_RX_PUSH_DRISC, which writes to DRISC L1): one
// 4 KiB packet plus header (4352 B), small enough to leave all four 2D VC0 sender channels at 16 slots (80 slots
// total; 8928 B drops slot capacity to 79 and halves the forwarding sender channels to 8 slots).
constexpr uint32_t PHASE_PROFILE_LOOPBACK_SCRATCH_SIZE = 4352;

// Host side allocates only the regions needed by the active flags.
constexpr uint32_t phase_profile_buffer_bytes(
    uint32_t mode, uint32_t exp_flags, bool txq_counters, bool stack_paint, uint32_t drisc_mode = 0) {
    const bool needs_profile = mode != 0 || txq_counters || stack_paint || drisc_mode != 0;
    const bool needs_loopback = (exp_flags & (EXP_RX_LOCAL_AS_PUSH | EXP_RX_PUSH_DRISC)) == EXP_RX_LOCAL_AS_PUSH;
    if (mode >= 3) {
        return PHASE_PROFILE_BUFFER_SIZE + PHASE_PROFILE_MICROBENCH_SIZE;
    }
    if (needs_profile) {
        return PHASE_PROFILE_BUFFER_SIZE + (needs_loopback ? PHASE_PROFILE_LOOPBACK_SCRATCH_SIZE : 0);
    }
    return needs_loopback ? PHASE_PROFILE_LOOPBACK_SCRATCH_SIZE : 0;
}
