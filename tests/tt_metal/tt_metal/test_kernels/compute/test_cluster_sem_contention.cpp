// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Contention probes for the Quasar cluster semaphore registers, inside one cluster. Every TRISC of every NEO
// runs this kernel. Participant p = NEO * 4 + TRISC. The host picks the active participants with a bit mask,
// so it also picks the placement (all on one NEO, or spread over the NEOs).
//
// A poll of a cluster register is load on the block under test. Thus inactive participants do not wait in
// sync_threads() (which polls semaphore 29) while others measure. They wait on an L1 flag, then join it.
//
// Modes:
//  - THROUGHPUT: each active participant hammers its semaphore until `window` ticks pass. Up to `depth`
//    accesses are issued before the first result is used, so one TRISC can offer more than one access per
//    round trip if its load unit keeps several reads in flight.
//  - VICTIM: participant 0 times single posts on semaphore 0 into a histogram, while the other active
//    participants run THROUGHPUT.
//  - BARRIER: the active participants run `rounds` back-to-back tensix_global_sem_barrier() calls.
//
// perf_ctrl != 0 (THROUGHPUT and ACCUM): TRISC 0 of each NEO, if active, programs its NEO's L1 client event
// counter with that GROUP_PERF_CTRL value around its own run and reports the count and the wall-clock window.

#include <cstdint>

#include "api/compute/common.h"
#include "api/kernel_thread_globals.h"
#include "ckernel_risc_atomics.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "internal/hw_thread.h"
#include "risc_common.h"

namespace {

constexpr uint32_t kModeThroughput = 0;
constexpr uint32_t kModeVictim = 1;
constexpr uint32_t kModeBarrier = 2;
constexpr uint32_t kModeAccum = 3;        // Pattern D: each active TRISC issues `rounds` atomic up(1)s
constexpr uint32_t kModeSubBankLoad = 4;  // 16-TRISC local L1-atomic sub-bank loader for cross-cluster tests
constexpr uint32_t kOpRead = 0;

// Report layout, in words. The host mirrors it in test_sem_scope.cpp.
constexpr uint32_t kMaxParticipants = 16;
constexpr uint32_t kRecordWords = 16;
constexpr uint32_t kHistBuckets = 64;
constexpr uint32_t kHistBucketTicks = 4;
constexpr uint32_t kL1SemBaseWord = 352;   // 64-byte aligned start of the 16 L1 semaphore slots
constexpr uint32_t kL1SemStrideWords = 4;  // 16 bytes per semaphore slot (one L1 sub-bank word)

// Cross-cluster shared offsets (mirrors test_cross_cluster_sem.cpp for kModeSubBankLoad).
constexpr uint32_t kCrossStopOffset = 1040;
constexpr uint32_t kCrossLoadDoneOffset = 1056;
constexpr uint32_t kCrossSubBankLoadOffset = 1536;  // kL1CounterOffset (1024) + 512: same L1 sub-bank
enum Record : uint32_t {
    ACCESSES = 0,
    START = 1,
    END = 2,
    MAX_SEEN = 3,   // largest pre-post count returned to this participant
    ZERO_GETS = 4,  // gets that returned a count of 0; must stay 0
    SAMPLES = 5,    // VICTIM only
    LAT_MIN = 6,
    LAT_MAX = 7,
    LAT_SUM = 8,
    PERF_COUNT = 9,   // L1 client event count over this participant's run (perf_ctrl != 0 only)
    PERF_TICKS = 10,  // wall-clock ticks between arming and reading the counter
};

// This NEO's L1 client event counter, through the TRISC local register window.
constexpr uint32_t kL1ClientPerfCtrl = LOCAL_REGS_BASE + (NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR -
                                                          NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_REG_FILE_BASE_ADDR);
constexpr uint32_t kL1ClientPerfCnt = LOCAL_REGS_BASE + (NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CNT_REG_ADDR -
                                                         NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_REG_FILE_BASE_ADDR);

// Selects the subport and event and clears the count (a read clears it). Returns the window start.
inline uint32_t l1_client_counter_arm(uint32_t ctrl) {
    *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(kL1ClientPerfCtrl) = ctrl;
    (void)*reinterpret_cast<volatile uint32_t tt_reg_ptr*>(kL1ClientPerfCnt);
    return get_timestamp_32b();
}

// Reads the count, then disables the counter.
inline uint32_t l1_client_counter_read() {
    const uint32_t count = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(kL1ClientPerfCnt);
    *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(kL1ClientPerfCtrl) = 0;
    return count;
}

constexpr uint32_t kAccessesPerClockCheck = 32;
constexpr uint32_t kBarrierArrivedIdx = 1;
constexpr uint32_t kBarrierGenerationIdx = 2;

template <bool use_l1>
__attribute__((always_inline)) inline uint32_t sem_read(volatile uint32_t* l1_base, uint32_t idx) {
    if constexpr (use_l1) {
        return load_acquire(l1_base + idx * kL1SemStrideWords);
    } else {
        return tensix_global_sem_read(idx);
    }
}

template <bool use_l1>
__attribute__((always_inline)) inline uint32_t sem_post(volatile uint32_t* l1_base, uint32_t idx, uint32_t inc) {
    if constexpr (use_l1) {
        return __atomic_fetch_add(l1_base + idx * kL1SemStrideWords, inc, __ATOMIC_RELAXED);
    } else {
        return tensix_global_sem_fetch_add(idx, inc);
    }
}

template <bool use_l1>
__attribute__((always_inline)) inline uint32_t sem_get(volatile uint32_t* l1_base, uint32_t idx, uint32_t dec) {
    if constexpr (use_l1) {
        return __atomic_fetch_add(l1_base + idx * kL1SemStrideWords, 0u - dec, __ATOMIC_RELAXED);
    } else {
        return tensix_global_sem_fetch_sub(idx, dec);
    }
}

template <bool use_l1>
__attribute__((always_inline)) inline void sem_init(volatile uint32_t* l1_base, uint32_t idx, uint32_t value) {
    if constexpr (use_l1) {
        store_release(l1_base + idx * kL1SemStrideWords, value);
    } else {
        tensix_global_sem_init(idx, value);
    }
}

inline void l1_atomic_sem_barrier(
    volatile uint32_t* arrived_ptr, volatile uint32_t* generation_ptr, uint32_t participants) {
    if (participants <= 1) {
        return;
    }
    asm volatile("fence rw, rw" ::: "memory");
    const uint32_t next_generation = (load_acquire(generation_ptr) + 1) & TENSIX_GLOBAL_SEM_VALUE_MASK;
    const uint32_t arrived = __atomic_fetch_add(arrived_ptr, 1u, __ATOMIC_ACQ_REL) + 1;
    if (arrived == participants) {
        store_release(arrived_ptr, uint32_t{0});
        store_release(generation_ptr, next_generation);
    } else {
        while ((load_acquire(generation_ptr) & TENSIX_GLOBAL_SEM_VALUE_MASK) != next_generation) {
        }
    }
    asm volatile("fence rw, rw" ::: "memory");
}

// Issues `depth` accesses before it uses any result. For posts, every result is used before the gets are
// issued: the in-order core then cannot issue a get until all of this batch's posts have returned, so a get
// never meets a count without its own post in it.
template <uint32_t depth, bool use_l1>
__attribute__((always_inline)) inline void batch(
    volatile uint32_t* l1_base, uint32_t idx, uint32_t op, uint32_t& max_seen, uint32_t& zero_gets) {
    uint32_t v[depth];
    if (op == kOpRead) {
#pragma GCC unroll 8
        for (uint32_t i = 0; i < depth; i++) {
            v[i] = sem_read<use_l1>(l1_base, idx);
        }
#pragma GCC unroll 8
        for (uint32_t i = 0; i < depth; i++) {
            const uint32_t count = v[i] & TENSIX_GLOBAL_SEM_VALUE_MASK;
            max_seen = count > max_seen ? count : max_seen;
        }
        return;
    }
#pragma GCC unroll 8
    for (uint32_t i = 0; i < depth; i++) {
        v[i] = sem_post<use_l1>(l1_base, idx, 1);
    }
#pragma GCC unroll 8
    for (uint32_t i = 0; i < depth; i++) {
        const uint32_t count = v[i] & TENSIX_GLOBAL_SEM_VALUE_MASK;
        max_seen = count > max_seen ? count : max_seen;
    }
    asm volatile("" ::"r"(max_seen) : "memory");
#pragma GCC unroll 8
    for (uint32_t i = 0; i < depth; i++) {
        v[i] = sem_get<use_l1>(l1_base, idx, 1);
    }
#pragma GCC unroll 8
    for (uint32_t i = 0; i < depth; i++) {
        zero_gets += (v[i] & TENSIX_GLOBAL_SEM_VALUE_MASK) == 0;
    }
}

template <uint32_t depth, bool use_l1>
void hammer(
    volatile uint32_t* l1_base, uint32_t idx, uint32_t op, uint32_t window, volatile tt_l1_ptr uint32_t* record) {
    const uint32_t accesses_per_batch = op == kOpRead ? depth : 2 * depth;
    const uint32_t batches = kAccessesPerClockCheck / accesses_per_batch;
    uint32_t accesses = 0;
    uint32_t max_seen = 0;
    uint32_t zero_gets = 0;
    const uint32_t start = get_timestamp_32b();
    uint32_t now;
    do {
        for (uint32_t b = 0; b < batches; b++) {
            batch<depth, use_l1>(l1_base, idx, op, max_seen, zero_gets);
        }
        accesses += batches * accesses_per_batch;
        now = get_timestamp_32b();
    } while (now - start < window);
    record[ACCESSES] = accesses;
    record[START] = start;
    record[END] = now;
    record[MAX_SEEN] = max_seen;
    record[ZERO_GETS] = zero_gets;
}

// The `mv` consumes its input, so the in-order core issues nothing after it until that load returns.
__attribute__((always_inline)) inline void wait_for(uint32_t value) {
    uint32_t sink;
    asm volatile("mv %0, %1" : "=r"(sink) : "r"(value) : "memory");
}

// Each sample includes one wall-clock read; the host compares against the run with no other participants.
template <bool use_l1>
void victim(
    volatile uint32_t* l1_base,
    uint32_t window,
    volatile tt_l1_ptr uint32_t* record,
    volatile tt_l1_ptr uint32_t* hist) {
    uint32_t samples = 0;
    uint32_t lat_min = ~0u;
    uint32_t lat_max = 0;
    uint32_t lat_sum = 0;
    uint32_t max_seen = 0;
    uint32_t zero_gets = 0;
    const uint32_t start = get_timestamp_32b();
    uint32_t t1;
    do {
        const uint32_t t0 = get_timestamp_32b();
        wait_for(t0);
        const uint32_t old = sem_post<use_l1>(l1_base, 0, 1);
        wait_for(old);
        t1 = get_timestamp_32b();
        const uint32_t lat = t1 - t0;
        const uint32_t count = old & TENSIX_GLOBAL_SEM_VALUE_MASK;
        max_seen = count > max_seen ? count : max_seen;
        zero_gets += (sem_get<use_l1>(l1_base, 0, 1) & TENSIX_GLOBAL_SEM_VALUE_MASK) == 0;
        const uint32_t bucket = lat / kHistBucketTicks;
        hist[bucket < kHistBuckets ? bucket : kHistBuckets - 1] += 1;
        lat_min = lat < lat_min ? lat : lat_min;
        lat_max = lat > lat_max ? lat : lat_max;
        lat_sum += lat;
        samples++;
    } while (t1 - start < window);
    record[ACCESSES] = 2 * samples;
    record[START] = start;
    record[END] = t1;
    record[MAX_SEEN] = max_seen;
    record[ZERO_GETS] = zero_gets;
    record[SAMPLES] = samples;
    record[LAT_MIN] = lat_min;
    record[LAT_MAX] = lat_max;
    record[LAT_SUM] = lat_sum;
}

template <bool use_l1>
void barrier(volatile uint32_t* l1_base, uint32_t participants, uint32_t rounds, volatile tt_l1_ptr uint32_t* record) {
    const uint32_t start = get_timestamp_32b();
    for (uint32_t r = 0; r < rounds; r++) {
        if constexpr (use_l1) {
            l1_atomic_sem_barrier(
                l1_base + kBarrierArrivedIdx * kL1SemStrideWords,
                l1_base + kBarrierGenerationIdx * kL1SemStrideWords,
                participants);
        } else {
            tensix_global_sem_barrier(kBarrierArrivedIdx, kBarrierGenerationIdx, participants);
        }
    }
    record[END] = get_timestamp_32b();
    record[START] = start;
    record[ACCESSES] = rounds;
}

// Pattern D (from bh-dm-compute-atomics): each active participant issues `rounds` atomic up(1)s on `idx`
// without decrementing, so the final counter value must equal `holders * rounds` with zero lost updates.
template <bool use_l1>
void accum(volatile uint32_t* l1_base, uint32_t idx, uint32_t rounds, volatile tt_l1_ptr uint32_t* record) {
    uint32_t max_seen = 0;
    const uint32_t start = get_timestamp_32b();
    for (uint32_t r = 0; r < rounds; r++) {
        const uint32_t old = sem_post<use_l1>(l1_base, idx, 1) & TENSIX_GLOBAL_SEM_VALUE_MASK;
        max_seen = old > max_seen ? old : max_seen;
    }
    record[END] = get_timestamp_32b();
    record[START] = start;
    record[ACCESSES] = rounds;
    record[MAX_SEEN] = max_seen;
    record[ZERO_GETS] = 0;
}

template <bool use_l1>
void run_active(
    uint32_t mode,
    uint32_t op,
    uint32_t depth,
    uint32_t idx,
    uint32_t p,
    uint32_t active_mask,
    uint32_t window,
    uint32_t rounds,
    volatile uint32_t* l1_base,
    volatile tt_l1_ptr uint32_t* record,
    volatile tt_l1_ptr uint32_t* hist) {
    if (mode == kModeBarrier) {
        barrier<use_l1>(l1_base, __builtin_popcount(active_mask), rounds, record);
    } else if (mode == kModeAccum) {
        accum<use_l1>(l1_base, idx, rounds, record);
    } else if (mode == kModeVictim && p == 0) {
        victim<use_l1>(l1_base, window, record, hist);
    } else {
        switch (depth) {
            case 1: hammer<1, use_l1>(l1_base, idx, op, window, record); break;
            case 2: hammer<2, use_l1>(l1_base, idx, op, window, record); break;
            case 4: hammer<4, use_l1>(l1_base, idx, op, window, record); break;
            case 8: hammer<8, use_l1>(l1_base, idx, op, window, record); break;
            default: ASSERT(false);
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t mode = get_arg(args::mode);
    const uint32_t op = get_arg(args::op);
    const uint32_t depth = get_arg(args::depth);
    const uint32_t distinct = get_arg(args::distinct);
    const uint32_t active_mask = get_arg(args::active_mask);
    const uint32_t window = get_arg(args::window);
    const uint32_t rounds = get_arg(args::rounds);
    const uint32_t report_addr = get_arg(args::report_addr);
    const bool use_l1 = get_arg(args::use_l1) != 0;
    const uint32_t perf_ctrl = get_arg(args::perf_ctrl);

    const uint32_t p = get_my_thread_id() * NUM_TRISC_CORES + internal_::get_trisc_id();
    const bool active = ((active_mask >> p) & 1) != 0;

    // Cross-cluster background load mode: all active TRISCs hammer kCrossSubBankLoadOffset (same L1 sub-bank as
    // the cross-cluster L1 counter) via NEO L1 port atomics until the DM coordinator sets kCrossStopOffset.
    // Never calls sync_threads(), so tt_cluster_sync is completely idle while the DM test runs.
    if (mode == kModeSubBankLoad) {
        if (!active) {
            return;
        }
        auto* target = reinterpret_cast<volatile uint32_t*>(report_addr + kCrossSubBankLoadOffset);
        auto* stop =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + kCrossStopOffset + MEM_L1_UNCACHED_BASE);
        uint32_t accesses = 0;
        uint32_t max_seen = 0;
        uint32_t zero_gets = 0;
        while (*stop == 0) {
            for (uint32_t b = 0; b < kAccessesPerClockCheck / 8; b++) {
                batch<4, true>(target, 0, op, max_seen, zero_gets);
            }
            accesses += kAccessesPerClockCheck;
        }
        if (p == static_cast<uint32_t>(__builtin_ctz(active_mask))) {
            auto* done_ops = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                report_addr + kCrossLoadDoneOffset + sizeof(uint32_t) + MEM_L1_UNCACHED_BASE);
            *done_ops = accesses;
        }
        return;
    }

    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    auto* l1_base = reinterpret_cast<volatile uint32_t*>(report_addr + kL1SemBaseWord * sizeof(uint32_t));
    volatile tt_l1_ptr uint32_t* record = report + p * kRecordWords;
    volatile tt_l1_ptr uint32_t* done_flag = report + kMaxParticipants * kRecordWords;
    volatile tt_l1_ptr uint32_t* hist = done_flag + 1;
    volatile tt_l1_ptr uint32_t* final_values = hist + kHistBuckets;

    sync_threads();
    if (active) {
        const uint32_t idx = distinct ? p : 0;
        const bool perf =
            perf_ctrl != 0 && internal_::get_trisc_id() == 0 && (mode == kModeThroughput || mode == kModeAccum);
        const uint32_t perf_start = perf ? l1_client_counter_arm(perf_ctrl) : 0;
        if (use_l1) {
            run_active<true>(mode, op, depth, idx, p, active_mask, window, rounds, l1_base, record, hist);
        } else {
            run_active<false>(mode, op, depth, idx, p, active_mask, window, rounds, l1_base, record, hist);
        }
        if (perf) {
            record[PERF_COUNT] = l1_client_counter_read();
            record[PERF_TICKS] = get_timestamp_32b() - perf_start;
        }
        if (p == static_cast<uint32_t>(__builtin_ctz(active_mask))) {
            *done_flag = 1;
        }
    } else {
        while (*done_flag == 0) {
        }
    }
    sync_threads();

    // Report the end counts, then leave every semaphore this kernel can touch at 0 (INIT also clears errors).
    if (p == 0) {
        for (uint32_t i = 0; i < kMaxParticipants; i++) {
            if (use_l1) {
                final_values[i] = sem_read<true>(l1_base, i) & TENSIX_GLOBAL_SEM_VALUE_MASK;
                sem_init<true>(l1_base, i, 0);
            } else {
                final_values[i] = sem_read<false>(l1_base, i) & TENSIX_GLOBAL_SEM_VALUE_MASK;
                sem_init<false>(l1_base, i, 0);
            }
        }
    }
}
