// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-cluster counter probes: a cluster semaphore on the home cluster versus an L1 word on the home cluster.
// The same kernel runs on the home cluster and on the remote cluster, one participant per DM thread.
// Home DM thread 0 is the coordinator and never a client. Every client reaches the counter over the NoC, also on
// the home cluster (loopback), so the two counters are compared on the same path. BARRIER2_GS is the exception:
// it uses register accesses inside each cluster and the NoC only between clusters.
//
// Semaphore access rules from the RTL: only 4-byte NoC reads and writes. A wider read of register space also
// reads +4, which is a GET of 1. NoC atomics must not target register space.
//   read  +0             plain read
//   read  +4*d (d=1..7)  GET d, returns the old value
//   read  +4*(i+8)       POST i (0..7), returns the old value
//   write +0             INIT
//   write +4             POST by the write data, no value returned
// The L1 counter is only touched by NoC atomics while clients run. DM AMOs and plain writes to its 16-byte chunk
// are not safe at the same time.
//
// Clocks are not synchronized across clusters, so every figure is an interval on one core (rdcycle).
//
// Background load (TestCrossClusterSemUnderLoad): DM threads in load_mask are loaders, not clients. A loader
// streams NoC writes into the home L1 at load_dst until the coordinator sets the stop flag on its node. A compute
// kernel on the home cluster may also run a pack/unpack datacopy; it writes its done word at kLoadDoneOffset.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

namespace {

enum Mode : uint32_t {
    RT_GS = 0,
    RT_L1 = 1,
    POSTED_GS = 2,
    POSTED_L1 = 3,
    BARRIER_GS = 4,
    BARRIER_L1 = 5,
    BARRIER2_GS = 6,
    POSTED_GS_STATEFUL = 7,
    BARRIER2_GS_PAUSED = 8,
    BARRIER2_GS_L1REL = 9,
    BARRIER2_GS_MCAST = 10,
    BARRIER2_L1 = 11,
    MCAST_GS = 12,
    MCAST_GS_STATEFUL = 13,
    MCAST_L1 = 14,
    HANDOFF_T1_GS = 15,  // Test 1: remote posted write (+4) -> local S4 poll -> L1 ack
    HANDOFF_T1_L1 = 16,  // Test 1: remote posted NoC atomic -> local uncached L1 poll -> L1 ack
    HANDOFF_T2_GS = 17,  // Test 2: local S4 post (+4) -> remote NoC read (+0) -> L1 ack
    HANDOFF_T2_L1 = 18,  // Test 2: local L1 atomic -> remote NoC L1 read -> L1 ack
    HANDOFF_T3_GS = 19,  // Test 3: remote posted write (+4) -> remote NoC read (+0) until visible
    HANDOFF_T3_L1 = 20,  // Test 3: remote posted NoC atomic -> remote NoC L1 read until visible
};

enum LoadKind : uint32_t {
    LOAD_NOC_TILE = 0,    // 2 KB NoC writes into home L1 (NIU slave-write pipeline)
    LOAD_S4_POLL = 1,     // Home DM loaders poll kAggressorSem + tid over the serial S4 APB port
    LOAD_L1_SUBBANK = 2,  // NoC atomics + uncached reads on the same L1 sub-bank / Flex-port 58 as l1_counter
    LOAD_S4_WRITE = 3,    // Home DM loaders hammer posted local writes into kAggressorSem + tid over S4
    LOAD_S4_RW = 4,       // Home DM loaders alternate local writes and reads into kAggressorSem + tid over S4
};

constexpr uint32_t kCounterSem = 0;
constexpr uint32_t kStartSem = 1;
constexpr uint32_t kEndSem = 2;
// BARRIER2_GS, on every cluster that has clients: local arrivals and the release count from the home coordinator.
constexpr uint32_t kLocalSem = 3;
constexpr uint32_t kReleaseSem = 4;
constexpr uint32_t kCounterSem2 = 5;   // second semaphore for separate-counter two-producer extension
constexpr uint32_t kAggressorSem = 6;  // kAggressorSem + tid (6..11) for local S4 read/write aggressors

// Report layout, in bytes from report_addr. The host mirrors it in test_sem_scope.cpp.
constexpr uint32_t kMaxThreads = 6;
constexpr uint32_t kRecordWords = 16;
enum Record : uint32_t {
    OPS = 0,
    CYCLES = 1,
    LAT_MAX = 2,
    MAX_SEEN = 3,
    ZERO_GETS = 4,
    TIMEOUT = 5,
    LOAD_OPS = 6,
    LAT_MIN = 7,
    LAT_SUM = 8,
    READ_ATTEMPTS = 9,
};
constexpr uint32_t kHomeOffset = kMaxThreads * kRecordWords * sizeof(uint32_t);
enum Home : uint32_t {
    FINAL_GS = 0,
    FINAL_L1 = 1,
    DRAIN_CYCLES = 2,
    HOME_TIMEOUT = 3,
    LOAD_DONE_AT_END = 4,
    REMOTE_FINAL_GS = 5,
    REMOTE_FINAL_L1 = 6,
    FINAL_GS2 = 7,
    FINAL_L12 = 8,
    PERF_COUNT_NEO0 = 9,  // L1 client event counts of home NEOs 0..3 (words 9..12), perf_ctrl only
    PERF_NEO_TICKS = 13,  // NEO wall-clock ticks between arming and reading the counters
    PERF_DM_CYCLES = 14,  // coordinator rdcycle over the same interval
};
constexpr uint32_t kScratchOffset = 512;       // one 16-byte NoC read return slot per thread
constexpr uint32_t kAckOffset = 640;           // two 16-byte L1 acknowledgment words (640 and 656)
constexpr uint32_t kL1CounterOffset = 1024;    // alone in its 16-byte chunk (cache line 16)
constexpr uint32_t kStopOffset = 1040;         // loader stop flag, on each node
constexpr uint32_t kLoadDoneOffset = 1056;     // the compute load's report; word 0 is nonzero once it finished
constexpr uint32_t kL1Counter2Offset = 1088;   // second L1 counter for separate-counter two-producer (cache line 17)
constexpr uint32_t kLocalL1Offset = 1152;      // per-cluster cached L1 arrival word for BARRIER2_L1 (cache line 18)
constexpr uint32_t kReleaseL1Offset = 1216;    // per-cluster uncached L1 release word (cache line 19)
constexpr uint32_t kSubBankLoadOffset = 1536;  // kL1CounterOffset + 512: same L1 sub-bank (cache line 24)
constexpr uint32_t kHistOffset = 1600;         // two 56-bucket latency histograms (1600..2048)
constexpr uint32_t kHandoffHistBuckets = 56;
constexpr uint32_t kHandoffHistBucketCycles = 16;
constexpr uint32_t kLoadSrcOffset = 2048;  // loader source; its contents do not matter
constexpr uint32_t kLoadBytes = 2048;      // one Float16_b tile
constexpr uint32_t kLoadBatch = 4;         // writes in flight before each write barrier

constexpr uint32_t kSpinCap = 1u << 22;
// The coordinator's local register reads share the home port with incoming NoC requests, so it polls sparsely.
constexpr uint32_t kPollGapCycles = 512;
// Short backoff between S4 reads in BARRIER2_GS_PAUSED / BARRIER2_GS_MCAST / BARRIER2_GS_L1REL so DM reads do not
// continuously hold S4 and starve incoming NoC or local POST_WRs.
constexpr uint32_t kBarrierPollGapCycles = 32;
// poll_gap == 0: the coordinator skips the drain poll and waits for the end with this gap instead.
constexpr uint32_t kQuietGapCycles = 1u << 16;

inline volatile tt_l1_ptr uint32_t* uncached(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
}

inline void pause(uint32_t gap) {
    const uint32_t t0 = get_timestamp_32b();
    while (get_timestamp_32b() - t0 < gap) {
    }
}

// L1 client event counters of the home NEOs, through each NEO's register window. The host packs one
// GROUP_PERF_CTRL value per NEO into 16 bits of perf_ctrl_lo (NEOs 0, 1) and perf_ctrl_hi (NEOs 2, 3); 0 = off.
constexpr uint32_t kNeoCount = 4;
constexpr uint32_t kNeoRegsStride =
    NEO_REGS_1__LOCAL_REGS_DEBUG_REGS_REG_FILE_BASE_ADDR - NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_REG_FILE_BASE_ADDR;

inline volatile uint32_t tt_reg_ptr* neo_reg(uint32_t neo, uint32_t neo0_addr) {
    return reinterpret_cast<volatile uint32_t tt_reg_ptr*>(neo0_addr + neo * kNeoRegsStride);
}

inline uint32_t neo_wall_clock() { return *neo_reg(0, NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_0_REG_ADDR); }

struct NeoCounters {
    uint32_t ctrl[kNeoCount];
    uint32_t neo_start = 0;
    uint32_t dm_start = 0;

    NeoCounters(uint32_t lo, uint32_t hi) : ctrl{lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16} {}
    bool enabled() const { return (ctrl[0] | ctrl[1] | ctrl[2] | ctrl[3]) != 0; }

    // Selects each NEO's subport and event and clears its count (a read clears it).
    void arm() {
        for (uint32_t n = 0; n < kNeoCount; n++) {
            if (ctrl[n] != 0) {
                *neo_reg(n, NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR) = ctrl[n];
                (void)*neo_reg(n, NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CNT_REG_ADDR);
            }
        }
        neo_start = neo_wall_clock();
        dm_start = get_timestamp_32b();
    }

    void read(volatile tt_l1_ptr uint32_t* out) const {
        const uint32_t dm_cycles = get_timestamp_32b() - dm_start;
        const uint32_t neo_ticks = neo_wall_clock() - neo_start;
        for (uint32_t n = 0; n < kNeoCount; n++) {
            uint32_t count = 0;
            if (ctrl[n] != 0) {
                count = *neo_reg(n, NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CNT_REG_ADDR);
                *neo_reg(n, NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR) = 0;
            }
            out[PERF_COUNT_NEO0 + n] = count;
        }
        out[PERF_NEO_TICKS] = neo_ticks;
        out[PERF_DM_CYCLES] = dm_cycles;
    }
};

// Programs the per-hart simple command buffer once for posted 4-byte unicast inline writes.
// Avoids both the 6 per-call register writes of noc_inline_dw_write and the shared-BSS
// noc_posted_writes_num_issued[noc] increment that causes L1 D$ coherence ping-pong across harts.
inline void scmdbuf_prep_unicast_post(uint64_t dest_noc_addr) {
    constexpr uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_POSTED;
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, NOC_UNICAST_WRITE_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, static_cast<uint32_t>(dest_noc_addr));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        static_cast<uint32_t>(dest_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
}

// Programs the per-hart simple command buffer for posted 4-byte multicast inline writes with source loopback.
// Unlinked single-packet flits avoid holding multicast path reservation across harts.
inline void scmdbuf_prep_mcast_post(uint64_t mcast_noc_addr, uint32_t num_dests) {
    constexpr uint64_t misc =
        CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_SRC_INCLUDE | CMD_BUF_MISC_POSTED;
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, NOC_MULTICAST_WRITE_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_MCAST_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, static_cast<uint32_t>(mcast_noc_addr));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        static_cast<uint32_t>(mcast_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
}

// Clears the sticky inline_wr=1 bit left in scmdbuf MISC by __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans.
inline void scmdbuf_restore_atomic() { init_at_cmd_buf(noc_local_xy(), MEM_NOC_ATOMIC_RET_VAL_ADDR); }

struct Target {
    uint32_t x;
    uint32_t y;
    uint32_t l1_counter;
    uint32_t scratch;

    uint64_t sem(uint32_t idx, uint32_t offset) const {
        return get_noc_addr(x, y, TENSIX_GLOBAL_SEM_BASE + idx * TENSIX_GLOBAL_SEM_STRIDE + offset);
    }
    uint64_t mcast_sem(const Target& other, uint32_t idx, uint32_t offset) const {
        const uint32_t x0 = x < other.x ? x : other.x;
        const uint32_t y0 = y < other.y ? y : other.y;
        const uint32_t x1 = x > other.x ? x : other.x;
        const uint32_t y1 = y > other.y ? y : other.y;
        return get_noc_multicast_addr(x0, y0, x1, y1, TENSIX_GLOBAL_SEM_BASE + idx * TENSIX_GLOBAL_SEM_STRIDE + offset);
    }
    uint32_t read(uint64_t src) const {
        noc_async_read(src, scratch, sizeof(uint32_t));
        noc_async_read_barrier();
        return *uncached(scratch);
    }
    uint32_t sem_post(uint32_t idx, uint32_t inc) const { return read(sem(idx, 4 * (inc + 8))); }
    uint32_t sem_get(uint32_t idx, uint32_t dec) const { return read(sem(idx, 4 * dec)); }
    uint32_t sem_value(uint32_t idx) const { return read(sem(idx, 0)) & TENSIX_GLOBAL_SEM_VALUE_MASK; }
    uint64_t l1() const { return get_noc_addr(x, y, l1_counter); }
    uint64_t mcast_l1(const Target& other) const {
        const uint32_t x0 = x < other.x ? x : other.x;
        const uint32_t y0 = y < other.y ? y : other.y;
        const uint32_t x1 = x > other.x ? x : other.x;
        const uint32_t y1 = y > other.y ? y : other.y;
        return get_noc_multicast_addr(x0, y0, x1, y1, l1_counter);
    }
    void l1_add(uint32_t v) const {
        noc_semaphore_inc(l1(), v);
        noc_async_atomic_barrier();
    }
};

void client(
    uint32_t mode,
    uint32_t window,
    uint32_t count,
    uint32_t participants,
    const Target& home,
    const Target& peer,
    volatile tt_l1_ptr uint32_t* record) {
    uint32_t ops = 0;
    uint32_t lat_max = 0;
    uint32_t max_seen = 0;
    uint32_t zero_gets = 0;
    uint32_t timeout = 0;
    const uint32_t start = get_timestamp_32b();
    uint32_t now = start;
    switch (mode) {
        case RT_GS:
        case RT_L1:
            do {
                const uint32_t t0 = get_timestamp_32b();
                if (mode == RT_GS) {
                    const uint32_t old = home.sem_post(kCounterSem, 1) & TENSIX_GLOBAL_SEM_VALUE_MASK;
                    max_seen = old > max_seen ? old : max_seen;
                    zero_gets += (home.sem_get(kCounterSem, 1) & TENSIX_GLOBAL_SEM_VALUE_MASK) == 0;
                } else {
                    home.l1_add(1);
                    home.l1_add(static_cast<uint32_t>(-1));
                }
                now = get_timestamp_32b();
                lat_max = now - t0 > lat_max ? now - t0 : lat_max;
                ops += 2;
            } while (now - start < window);
            break;
        case POSTED_GS:
            for (uint32_t i = 0; i < count; i++) {
                noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(home.sem(kCounterSem, 4), 1);
            }
            now = get_timestamp_32b();
            noc_async_posted_writes_flushed();
            scmdbuf_restore_atomic();
            ops = count;
            break;
        case POSTED_GS_STATEFUL:
            scmdbuf_prep_unicast_post(home.sem(kCounterSem, 4));
            for (uint32_t i = 0; i < count; i++) {
                __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(1);
            }
            now = get_timestamp_32b();
            noc_async_posted_writes_flushed();
            scmdbuf_restore_atomic();
            ops = count;
            break;
        case POSTED_L1:
            for (uint32_t i = 0; i < count; i++) {
                noc_semaphore_inc<true>(home.l1(), 1);
            }
            now = get_timestamp_32b();
            noc_async_posted_writes_flushed();
            ops = count;
            break;
        case MCAST_GS: {
            const uint64_t mcast_addr = home.mcast_sem(peer, kCounterSem, 4);
            for (uint32_t i = 0; i < count; i++) {
                scmdbuf_prep_mcast_post(mcast_addr, 2);
                __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(1);
                noc_posted_writes_num_issued[noc_index] += 1;
            }
            now = get_timestamp_32b();
            noc_async_posted_writes_flushed();
            scmdbuf_restore_atomic();
            ops = count;
            break;
        }
        case MCAST_GS_STATEFUL: {
            const uint64_t mcast_addr = home.mcast_sem(peer, kCounterSem, 4);
            scmdbuf_prep_mcast_post(mcast_addr, 2);
            for (uint32_t i = 0; i < count; i++) {
                __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(1);
            }
            now = get_timestamp_32b();
            noc_async_posted_writes_flushed();
            scmdbuf_restore_atomic();
            ops = count;
            break;
        }
        case MCAST_L1: {
            const uint64_t mcast_addr = home.mcast_l1(peer);
            for (uint32_t i = 0; i < count; i++) {
                noc_semaphore_inc_multicast<true>(mcast_addr, 1, 2);
            }
            now = get_timestamp_32b();
            noc_async_posted_writes_flushed();
            ops = count;
            break;
        }
        case BARRIER_GS:
        case BARRIER_L1:
            for (uint32_t r = 1; r <= count && timeout == 0; r++) {
                const uint32_t target = r * participants;
                uint32_t seen = 0;
                if (mode == BARRIER_GS) {
                    // POST via read at +4*(1+8) returns the pre-increment count, so the last
                    // arriving participant already knows the barrier reached target without polling.
                    seen = (home.sem_post(kCounterSem, 1) & TENSIX_GLOBAL_SEM_VALUE_MASK) + 1;
                } else {
                    home.l1_add(1);
                }
                uint32_t spin = 0;
                while (seen < target) {
                    seen = mode == BARRIER_GS ? home.sem_value(kCounterSem) : home.read(home.l1());
                    if (++spin == kSpinCap) {
                        timeout = 1;
                        break;
                    }
                }
            }
            now = get_timestamp_32b();
            ops = count;
            break;
        default: timeout = 1;
    }
    record[OPS] = ops;
    record[CYCLES] = now - start;
    record[LAT_MAX] = lat_max;
    record[MAX_SEEN] = max_seen;
    record[ZERO_GETS] = zero_gets;
    record[TIMEOUT] = timeout;
}

// Spins on a local semaphore until it reaches `target`. Returns false at the spin cap.
bool spin_local_sem(uint32_t idx, uint32_t target, uint32_t gap = 0) {
    for (uint32_t spin = 0; spin < kSpinCap; spin++) {
        if ((tensix_global_sem_read(idx) & TENSIX_GLOBAL_SEM_VALUE_MASK) >= target) {
            return true;
        }
        if (gap != 0) {
            pause(gap);
        }
    }
    return false;
}

// Spins on an uncached local L1 word until it reaches `target`. Returns false at the spin cap.
bool spin_local_l1(uint32_t addr, uint32_t target) {
    for (uint32_t spin = 0; spin < kSpinCap; spin++) {
        if (*uncached(addr) >= target) {
            return true;
        }
    }
    return false;
}

// Two-level barrier, client side. Each round: arrive on this cluster's local counter; the cluster leader (lowest
// active thread) waits for its cluster, then sends one arrival to the home counter; everyone waits on the local
// release counter. Only the leaders' arrivals and the coordinator's releases cross the NoC.
//   BARRIER2_GS:        1-cycle POST_WR (+4) for local arrivals, tight S4 poll on kLocalSem and kReleaseSem.
//   BARRIER2_GS_PAUSED: same, with kBarrierPollGapCycles between S4 reads so DM reads do not starve S4 writes.
//   BARRIER2_GS_MCAST:  same as BARRIER2_GS_PAUSED, with a single multicast POST_WR release from the coordinator.
//   BARRIER2_GS_L1REL:  POST_WR (+4) for local arrivals, paused S4 poll by the leader only, L1 word for release.
//   BARRIER2_L1:        cached L2 AMO for local arrivals, posted NoC atomic to home L1, L1 word for release.
void client_two_level(
    uint32_t mode,
    uint32_t count,
    uint32_t local_n,
    bool is_home,
    bool leader,
    const Target& home,
    uint32_t report_addr,
    volatile tt_l1_ptr uint32_t* record) {
    uint32_t timeout = 0;
    const uint32_t sem_gap = mode == BARRIER2_GS ? 0 : kBarrierPollGapCycles;
    const bool l1_release = mode == BARRIER2_GS_L1REL || mode == BARRIER2_L1;
    volatile uint32_t* local_l1_arrived = reinterpret_cast<volatile uint32_t*>(report_addr + kLocalL1Offset);
    const uint32_t release_l1_addr = report_addr + kReleaseL1Offset;

    const uint32_t start = get_timestamp_32b();
    for (uint32_t r = 1; r <= count && timeout == 0; r++) {
        if (mode == BARRIER2_L1) {
            __atomic_add_fetch(local_l1_arrived, 1, __ATOMIC_ACQ_REL);
            if (leader) {
                uint32_t spin = 0;
                while (__atomic_load_n(local_l1_arrived, __ATOMIC_ACQUIRE) < r * local_n) {
                    if (++spin == kSpinCap) {
                        timeout = 1;
                        break;
                    }
                }
                noc_semaphore_inc<true>(home.l1(), 1);
            }
        } else {
            tensix_global_sem_post(kLocalSem, 1);
            if (leader) {
                timeout |= spin_local_sem(kLocalSem, r * local_n, sem_gap) ? 0 : 1;
                if (is_home) {
                    tensix_global_sem_post(kCounterSem, 1);
                } else {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(home.sem(kCounterSem, 4), 1);
                }
            }
        }
        if (l1_release) {
            timeout |= spin_local_l1(release_l1_addr, r) ? 0 : 1;
        } else {
            timeout |= spin_local_sem(kReleaseSem, r, sem_gap) ? 0 : 1;
        }
    }
    const uint32_t now = get_timestamp_32b();
    // One more local arrival marks this client done; the leader then leaves this cluster's counters at 0.
    if (mode == BARRIER2_L1) {
        __atomic_add_fetch(local_l1_arrived, 1, __ATOMIC_ACQ_REL);
        if (leader) {
            uint32_t spin = 0;
            while (__atomic_load_n(local_l1_arrived, __ATOMIC_ACQUIRE) < (count + 1) * local_n) {
                if (++spin == kSpinCap) {
                    timeout = 1;
                    break;
                }
            }
            *local_l1_arrived = 0;
            flush_l2_cache_line(report_addr + kLocalL1Offset);
            *uncached(release_l1_addr) = 0;
            noc_async_posted_writes_flushed();
        }
    } else {
        tensix_global_sem_post(kLocalSem, 1);
        if (leader) {
            timeout |= spin_local_sem(kLocalSem, (count + 1) * local_n, sem_gap) ? 0 : 1;
            tensix_global_sem_init(kLocalSem, 0);
            tensix_global_sem_init(kReleaseSem, 0);
            if (l1_release) {
                *uncached(release_l1_addr) = 0;
            }
            noc_async_posted_writes_flushed();
            scmdbuf_restore_atomic();
        }
    }
    record[OPS] = count;
    record[CYCLES] = now - start;
    record[TIMEOUT] = timeout;
}

// Background load on DM threads in load_mask until the coordinator sets this node's stop flag:
//   LOAD_NOC_TILE:   batches of tile-sized writes from this node's L1 into the home L1. ponytail: every loader
//                    writes the same home tile, so the load hits one set of L1 banks; spread `dst` over several
//                    tiles if bank placement turns out to matter.
//   LOAD_S4_POLL:    home DM loaders poll kAggressorSem + tid in a tight loop over the serial S4 APB port.
//   LOAD_L1_SUBBANK: loaders issue posted NoC atomics to kSubBankLoadOffset (same L1 sub-bank as kL1CounterOffset)
//                    and home loaders also issue uncached L1 reads through Flex-client port 58.
//   LOAD_S4_WRITE:   home DM loaders hammer posted local writes into kAggressorSem + tid over S4.
//   LOAD_S4_RW:      home DM loaders alternate local writes and reads into kAggressorSem + tid over S4.
void loader(
    uint32_t load_kind,
    bool is_home,
    uint32_t tid,
    uint32_t report_addr,
    const Target& home,
    uint32_t src,
    uint64_t dst,
    uint32_t stop_addr,
    volatile tt_l1_ptr uint32_t* record) {
    uint32_t writes = 0;
    const uint32_t aggr_sem = kAggressorSem + tid;
    const uint32_t start = get_timestamp_32b();
    if (load_kind == LOAD_S4_POLL) {
        while (*uncached(stop_addr) == 0) {
            if (is_home) {
                for (uint32_t i = 0; i < 16; i++) {
                    (void)tensix_global_sem_read(aggr_sem);
                }
                writes += 16;
            } else {
                pause(64);
                writes += 1;
            }
        }
    } else if (load_kind == LOAD_S4_WRITE) {
        while (*uncached(stop_addr) == 0) {
            if (is_home) {
                for (uint32_t i = 0; i < 16; i++) {
                    tensix_global_sem_post(aggr_sem, 1);
                }
                tensix_global_sem_init(aggr_sem, 0);
                writes += 17;
            } else {
                pause(64);
                writes += 1;
            }
        }
    } else if (load_kind == LOAD_S4_RW) {
        while (*uncached(stop_addr) == 0) {
            if (is_home) {
                for (uint32_t i = 0; i < 8; i++) {
                    tensix_global_sem_post(aggr_sem, 1);
                    (void)tensix_global_sem_read(aggr_sem);
                }
                tensix_global_sem_init(aggr_sem, 0);
                writes += 17;
            } else {
                pause(64);
                writes += 1;
            }
        }
    } else if (load_kind == LOAD_L1_SUBBANK) {
        const uint64_t subbank_noc = get_noc_addr(home.x, home.y, report_addr + kSubBankLoadOffset);
        while (*uncached(stop_addr) == 0) {
            for (uint32_t i = 0; i < 8; i++) {
                noc_semaphore_inc<true>(subbank_noc, 1);
            }
            if (is_home) {
                for (uint32_t i = 0; i < 8; i++) {
                    (void)*uncached(report_addr + kSubBankLoadOffset);
                }
            }
            noc_async_posted_writes_flushed();
            writes += 8;
        }
    } else {
        while (*uncached(stop_addr) == 0) {
            for (uint32_t i = 0; i < kLoadBatch; i++) {
                noc_async_write(src, dst, kLoadBytes);
            }
            noc_async_write_barrier();
            writes += kLoadBatch;
        }
    }
    record[LOAD_OPS] = writes;
    record[CYCLES] = get_timestamp_32b() - start;
}

// Waits until the local semaphore reaches `target`, or the spin cap. Returns false on timeout.
bool wait_local_sem(uint32_t idx, uint32_t target, uint32_t gap = kPollGapCycles) {
    for (uint32_t spin = 0; spin < kSpinCap; spin++) {
        if ((tensix_global_sem_read(idx) & TENSIX_GLOBAL_SEM_VALUE_MASK) >= target) {
            return true;
        }
        pause(gap);
    }
    return false;
}

inline bool is_two_level_mode(uint32_t mode) {
    return mode == BARRIER2_GS || mode == BARRIER2_GS_PAUSED || mode == BARRIER2_GS_L1REL ||
           mode == BARRIER2_GS_MCAST || mode == BARRIER2_L1;
}

inline bool is_handoff_mode(uint32_t mode) { return mode >= HANDOFF_T1_GS && mode <= HANDOFF_T3_L1; }

inline bool is_posted_drain_mode(uint32_t mode) {
    return mode == POSTED_GS || mode == POSTED_GS_STATEFUL || mode == POSTED_L1 || mode == MCAST_GS ||
           mode == MCAST_GS_STATEFUL || mode == MCAST_L1;
}

inline bool is_mcast_mode(uint32_t mode) { return mode == MCAST_GS || mode == MCAST_GS_STATEFUL || mode == MCAST_L1; }

// Handoff round-trip and visibility probes (Tests 1, 2, 3 and Two-Producer Extension):
//   HANDOFF_T1_GS / L1: Producer posts to home counter -> home Detector polls locally -> L1 ack to Producer.
//   HANDOFF_T2_GS / L1: Home Producer posts locally -> external Detector reads over NoC -> L1 ack to Home Producer.
//   HANDOFF_T3_GS / L1: Producer posts to home counter -> Producer reads home counter until update is visible.
// In Two-Producer shared-counter runs (separate_counters == false), Producer 0 increments by 1 (low byte) and
// Producer 1 increments by 256 (high byte) on the same counter so neither producer's progress masks the other.
void client_handoff(
    uint32_t mode,
    uint32_t count,
    bool is_home,
    uint32_t tid,
    uint32_t remote_count,
    bool separate_counters,
    const Target& home,
    const Target& peer,
    uint32_t report_addr,
    volatile tt_l1_ptr uint32_t* record) {
    const bool use_l1 = (mode == HANDOFF_T1_L1 || mode == HANDOFF_T2_L1 || mode == HANDOFF_T3_L1);
    const bool is_t1 = (mode == HANDOFF_T1_GS || mode == HANDOFF_T1_L1);
    const bool is_t2 = (mode == HANDOFF_T2_GS || mode == HANDOFF_T2_L1);
    uint32_t timeout = 0;
    uint32_t read_attempts = 0;

    // Detector role in Test 1 (on home cluster, tid 1 or 2) and Test 2 (on external cluster, tid 0).
    if ((is_t1 && is_home && (tid == 1 || tid == 2)) || (is_t2 && !is_home && tid == 0)) {
        const uint32_t det_idx = is_t1 ? (tid - 1) : 0u;
        const uint32_t sem_idx = (det_idx == 1 && separate_counters) ? kCounterSem2 : kCounterSem;
        const uint32_t l1_addr =
            report_addr + ((det_idx == 1 && separate_counters) ? kL1Counter2Offset : kL1CounterOffset);
        const uint32_t shift = (det_idx == 1 && !separate_counters) ? 8u : 0u;
        const uint32_t ack_addr = report_addr + kAckOffset + det_idx * 16;
        const bool ack_on_home = is_t1 && (det_idx == 1 && remote_count == 1);
        if (!ack_on_home) {
            const Target& ack_target = is_t1 ? peer : home;
            scmdbuf_prep_unicast_post(get_noc_addr(ack_target.x, ack_target.y, ack_addr));
        }
        const uint32_t start = get_timestamp_32b();
        for (uint32_t r = 1; r <= count && timeout == 0; r++) {
            uint32_t spin = 0;
            while (true) {
                uint32_t v;
                if (is_t1) {
                    v = use_l1 ? *uncached(l1_addr) : (tensix_global_sem_read(sem_idx) & TENSIX_GLOBAL_SEM_VALUE_MASK);
                } else {
                    v = use_l1 ? home.read(home.l1()) : home.sem_value(sem_idx);
                }
                read_attempts++;
                if (((v >> shift) & 0xFFu) >= r) {
                    break;
                }
                if (++spin == kSpinCap) {
                    timeout = 1;
                    break;
                }
            }
            if (ack_on_home) {
                *uncached(ack_addr) = r;
            } else {
                __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(r);
            }
        }
        const uint32_t now = get_timestamp_32b();
        noc_async_posted_writes_flushed();
        scmdbuf_restore_atomic();
        record[OPS] = count;
        record[CYCLES] = now - start;
        record[READ_ATTEMPTS] = read_attempts;
        record[TIMEOUT] = timeout;
        return;
    }

    // Producer / Timer role in Tests 1, 2, and 3.
    const uint32_t p_idx = is_t2 ? 0u : ((!is_home && tid == 0) ? 0u : 1u);
    const uint32_t sem_idx = (p_idx == 1 && separate_counters) ? kCounterSem2 : kCounterSem;
    const uint32_t l1_addr = report_addr + ((p_idx == 1 && separate_counters) ? kL1Counter2Offset : kL1CounterOffset);
    const uint64_t l1_noc = get_noc_addr(home.x, home.y, l1_addr);
    const uint32_t step = (p_idx == 1 && !separate_counters) ? 256u : 1u;
    const uint32_t shift = (p_idx == 1 && !separate_counters) ? 8u : 0u;
    const uint32_t ack_addr = report_addr + kAckOffset + p_idx * 16;
    volatile tt_l1_ptr uint32_t* hist = uncached(report_addr + kHistOffset) + p_idx * kHandoffHistBuckets;

    if (!use_l1 && !is_home) {
        scmdbuf_prep_unicast_post(home.sem(sem_idx, 4));
    }

    uint32_t lat_min = ~0u;
    uint32_t lat_max = 0;
    uint32_t lat_sum = 0;
    const uint32_t start = get_timestamp_32b();
    for (uint32_t r = 1; r <= count && timeout == 0; r++) {
        const uint32_t t0 = get_timestamp_32b();
        if (use_l1) {
            noc_semaphore_inc<true>(l1_noc, step);
        } else if (is_home) {
            tensix_global_sem_post(sem_idx, step);
        } else {
            __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(step);
        }

        if (is_t1 || is_t2) {
            if (!spin_local_l1(ack_addr, r)) {
                timeout = 1;
                break;
            }
        } else {
            uint32_t spin = 0;
            while (true) {
                uint32_t v;
                if (use_l1) {
                    v = is_home ? *uncached(l1_addr) : home.read(l1_noc);
                } else {
                    v = is_home ? (tensix_global_sem_read(sem_idx) & TENSIX_GLOBAL_SEM_VALUE_MASK)
                                : home.sem_value(sem_idx);
                }
                read_attempts++;
                if (((v >> shift) & 0xFFu) >= r) {
                    break;
                }
                if (++spin == kSpinCap) {
                    timeout = 1;
                    break;
                }
            }
        }
        const uint32_t t1 = get_timestamp_32b();
        const uint32_t lat = t1 - t0;
        const uint32_t bucket = lat / kHandoffHistBucketCycles;
        hist[bucket < kHandoffHistBuckets ? bucket : kHandoffHistBuckets - 1] += 1;
        lat_min = lat < lat_min ? lat : lat_min;
        lat_max = lat > lat_max ? lat : lat_max;
        lat_sum += lat;
    }
    const uint32_t now = get_timestamp_32b();
    noc_async_posted_writes_flushed();
    scmdbuf_restore_atomic();
    record[OPS] = count;
    record[CYCLES] = now - start;
    record[LAT_MIN] = lat_min;
    record[LAT_MAX] = lat_max;
    record[LAT_SUM] = lat_sum;
    record[READ_ATTEMPTS] = read_attempts;
    record[TIMEOUT] = timeout;
}

void coordinator(
    uint32_t mode,
    uint32_t count,
    uint32_t participants,
    uint32_t poll_gap,
    uint32_t home_count,
    uint32_t remote_count,
    const Target& home,
    const Target& peer,
    uint32_t report_addr,
    NeoCounters& counters) {
    volatile tt_l1_ptr uint32_t* out = uncached(report_addr + kHomeOffset);
    uint32_t timeout = 0;
    uint32_t drain = 0;
    // Armed before the start so the register accesses stay out of the clients' timed window.
    if (counters.enabled()) {
        counters.arm();
    }
    if (!wait_local_sem(kStartSem, participants)) {
        timeout = 1;
    }
    const uint32_t start = get_timestamp_32b();
    if (is_two_level_mode(mode)) {
        // One arrival per cluster with clients; release each such cluster through kReleaseSem or kReleaseL1Offset.
        const uint32_t clusters = (home_count != 0) + (remote_count != 0);
        const uint32_t sem_gap = mode == BARRIER2_GS ? 0 : kBarrierPollGapCycles;
        const bool l1_release = mode == BARRIER2_GS_L1REL || mode == BARRIER2_L1;
        const uint64_t mcast_release = home.mcast_sem(peer, kReleaseSem, 4);
        for (uint32_t r = 1; r <= count && timeout == 0; r++) {
            if (mode == BARRIER2_L1) {
                timeout |= spin_local_l1(home.l1_counter, r * clusters) ? 0 : 1;
            } else {
                timeout |= spin_local_sem(kCounterSem, r * clusters, sem_gap) ? 0 : 1;
            }
            if (l1_release) {
                if (remote_count != 0) {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(
                        get_noc_addr(peer.x, peer.y, report_addr + kReleaseL1Offset), r);
                }
                if (home_count != 0) {
                    *uncached(report_addr + kReleaseL1Offset) = r;
                }
            } else if (mode == BARRIER2_GS_MCAST && home_count != 0 && remote_count != 0) {
                scmdbuf_prep_mcast_post(mcast_release, 2);
                __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(1);
            } else {
                if (remote_count != 0) {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(peer.sem(kReleaseSem, 4), 1);
                }
                if (home_count != 0) {
                    tensix_global_sem_post(kReleaseSem, 1);
                }
            }
        }
        noc_async_posted_writes_flushed();
        scmdbuf_restore_atomic();
    }
    if (is_posted_drain_mode(mode) && poll_gap != 0) {
        const uint32_t expected = participants * count;
        const bool use_l1 = mode == POSTED_L1 || mode == MCAST_L1;
        bool done = false;
        for (uint32_t spin = 0; spin < kSpinCap && !done; spin++) {
            const uint32_t v = use_l1 ? *uncached(home.l1_counter)
                                      : (tensix_global_sem_read(kCounterSem) & TENSIX_GLOBAL_SEM_VALUE_MASK);
            done = v >= expected;
            if (!done) {
                pause(poll_gap);
            }
        }
        drain = get_timestamp_32b() - start;
        timeout |= done ? 0 : 1;
    }
    if (!wait_local_sem(kEndSem, participants, poll_gap != 0 ? poll_gap : kQuietGapCycles)) {
        timeout = 1;
    }
    // Read before the loaders stop, so the window covers only time with the load running.
    if (counters.enabled()) {
        counters.read(out);
    }
    // The clients are done: record whether the compute load outlasted them, then stop the loaders on both nodes.
    const uint32_t load_done = *uncached(report_addr + kLoadDoneOffset);
    *uncached(report_addr + kStopOffset) = 1;
    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(get_noc_addr(peer.x, peer.y, report_addr + kStopOffset), 1);
    noc_async_posted_writes_flushed();
    scmdbuf_restore_atomic();

    uint32_t remote_final_gs = 0;
    uint32_t remote_final_l1 = 0;
    if (is_mcast_mode(mode)) {
        const uint32_t expected = participants * count;
        for (uint32_t spin = 0; spin < kSpinCap; spin++) {
            const uint32_t rv = mode == MCAST_L1 ? peer.read(peer.l1()) : peer.sem_value(kCounterSem);
            if (rv >= expected) {
                break;
            }
        }
        remote_final_gs = peer.sem_value(kCounterSem);
        remote_final_l1 = peer.read(peer.l1());
        noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(peer.sem(kCounterSem, 0), 0);
        noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(peer.l1(), 0);
        noc_async_posted_writes_flushed();
        scmdbuf_restore_atomic();
    }

    // Report, then leave every counter at 0 for the next run.
    out[FINAL_GS] = tensix_global_sem_read(kCounterSem) & TENSIX_GLOBAL_SEM_VALUE_MASK;
    out[FINAL_L1] = *uncached(home.l1_counter);
    out[DRAIN_CYCLES] = drain;
    out[HOME_TIMEOUT] = timeout;
    out[LOAD_DONE_AT_END] = load_done;
    out[REMOTE_FINAL_GS] = remote_final_gs;
    out[REMOTE_FINAL_L1] = remote_final_l1;
    out[FINAL_GS2] = tensix_global_sem_read(kCounterSem2) & TENSIX_GLOBAL_SEM_VALUE_MASK;
    out[FINAL_L12] = *uncached(report_addr + kL1Counter2Offset);
    tensix_global_sem_init(kCounterSem, 0);
    tensix_global_sem_init(kCounterSem2, 0);
    tensix_global_sem_init(kStartSem, 0);
    tensix_global_sem_init(kEndSem, 0);
    for (uint32_t t = 0; t < kMaxThreads; t++) {
        tensix_global_sem_init(kAggressorSem + t, 0);
    }
    *uncached(home.l1_counter) = 0;
    *uncached(report_addr + kL1Counter2Offset) = 0;
    *uncached(report_addr + kSubBankLoadOffset) = 0;
    *uncached(report_addr + kAckOffset) = 0;
    *uncached(report_addr + kAckOffset + 16) = 0;
}

}  // namespace

void kernel_main() {
    const uint32_t mode = get_arg(args::mode);
    const uint32_t window = get_arg(args::window);
    const uint32_t count = get_arg(args::count);
    const uint32_t is_home = get_arg(args::is_home);
    const uint32_t active_mask = get_arg(args::active_mask);
    const uint32_t participants = get_arg(args::participants);
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t poll_gap = get_arg(args::poll_gap);
    const uint32_t home_count = get_arg(args::home_count);
    const uint32_t remote_count = get_arg(args::remote_count);
    const uint32_t load_mask = get_arg(args::load_mask);
    const uint32_t load_dst = get_arg(args::load_dst);
    const uint32_t load_kind = get_arg(args::load_kind);
    const bool separate_counters = get_arg(args::separate_counters) != 0;
    const uint32_t tid = get_my_thread_id();
    const Target home{
        .x = get_arg(args::home_noc_x),
        .y = get_arg(args::home_noc_y),
        .l1_counter = report_addr + kL1CounterOffset,
        .scratch = report_addr + kScratchOffset + tid * 16,
    };
    const Target peer{
        .x = get_arg(args::peer_noc_x),
        .y = get_arg(args::peer_noc_y),
        .l1_counter = report_addr + kL1CounterOffset,
        .scratch = report_addr + kScratchOffset + tid * 16,
    };

    if (is_home && tid == 0) {
        NeoCounters counters(get_arg(args::perf_ctrl_lo), get_arg(args::perf_ctrl_hi));
        coordinator(mode, count, participants, poll_gap, home_count, remote_count, home, peer, report_addr, counters);
        return;
    }
    volatile tt_l1_ptr uint32_t* record = uncached(report_addr) + tid * kRecordWords;
    if ((load_mask >> tid) & 1) {
        loader(
            load_kind,
            is_home != 0,
            tid,
            report_addr,
            home,
            report_addr + kLoadSrcOffset,
            get_noc_addr(home.x, home.y, load_dst),
            report_addr + kStopOffset,
            record);
        return;
    }
    if (((active_mask >> tid) & 1) == 0) {
        return;
    }

    // Start together: every client posts the start semaphore, then waits until all of them have.
    home.sem_post(kStartSem, 1);
    for (uint32_t spin = 0; home.sem_value(kStartSem) < participants; spin++) {
        if (spin == kSpinCap) {
            record[TIMEOUT] = 1;
            home.sem_post(kEndSem, 1);
            return;
        }
    }
    if (is_two_level_mode(mode)) {
        const bool leader = static_cast<uint32_t>(__builtin_ctz(active_mask)) == tid;
        client_two_level(
            mode, count, is_home ? home_count : remote_count, is_home != 0, leader, home, report_addr, record);
    } else if (is_handoff_mode(mode)) {
        client_handoff(
            mode, count, is_home != 0, tid, remote_count, separate_counters, home, peer, report_addr, record);
    } else {
        client(mode, window, count, participants, home, peer, record);
    }
    home.sem_post(kEndSem, 1);
}
