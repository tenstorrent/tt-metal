// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _DRAM_COMMON_H
#define _DRAM_COMMON_H

#include <stdint.h>

#include "patterns/patterns.hpp"

static constexpr uint32_t DRAM_TEST_NOC_WORD_BYTES = 64;
static constexpr uint64_t DRAM_TEST_MAX_BANK_BYTES = 0xFF000000ULL;
static constexpr uint64_t DRAM_TEST_RESERVED_TOP_BYTES = 2048ULL;
static constexpr uint64_t DRAM_TEST_EFFECTIVE_MAX_BANK_BYTES = DRAM_TEST_MAX_BANK_BYTES - DRAM_TEST_RESERVED_TOP_BYTES;
static constexpr uint32_t DRAM_TEST_BYTES = 32u * 1024u * 1024u;  // 32MB

// Recommended small default queue depth for persistent worker mode.
// You can override from host-side code if needed.
static constexpr uint32_t DRAM_JOB_QUEUE_CAPACITY = 8u;

// -------------------------
// Failure classification
// -------------------------
static constexpr uint32_t DRAM_FAILURE_NONE = 0u;
static constexpr uint32_t DRAM_FAILURE_WRITE = 1u;
static constexpr uint32_t DRAM_FAILURE_READ = 2u;

// -------------------------
// Persistent worker status
// -------------------------
static constexpr uint32_t DRAM_PROGRESS_MAGIC = 0x4452414Du;  // "DRAM"

static constexpr uint32_t DRAM_PROGRESS_STATE_IDLE = 0u;
static constexpr uint32_t DRAM_PROGRESS_STATE_RUNNING = 1u;
static constexpr uint32_t DRAM_PROGRESS_STATE_DONE = 2u;
static constexpr uint32_t DRAM_PROGRESS_STATE_ERROR = 3u;

static constexpr uint32_t DRAM_PROGRESS_STAGE_WAIT = 0u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_JOB_START = 1u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_PREPARE = 2u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_WRITE = 3u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_READ = 4u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_VERIFY = 5u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_REREAD = 6u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_JOB_DONE = 7u;
static constexpr uint32_t DRAM_PROGRESS_STAGE_DONE = 8u;

// -------------------------
// Original single-job result
// -------------------------
struct DramBaseResult {
    uint32_t job_id;
    uint32_t pattern_id;
    uint32_t pass_index;
    uint32_t repeat_index;
    uint32_t bank_id;
    uint32_t transfers;
    uint32_t words_checked;
    uint32_t failures;
    uint32_t first_fail_addr;
    uint32_t first_expected;
    uint32_t first_observed;

    uint32_t failure_kind;    // 0 none, 1 write, 2 read
    uint32_t readback_count;  // usually 5
    uint32_t readback_data[5];

    uint32_t suspected_write_failures;  // number of words classified as write errors
    uint32_t suspected_read_failures;   // number of words classified as read errors

    uint64_t prepare_ticks;
    uint64_t write_ticks;
    uint64_t read_ticks;

    // Profiler: finer-grained timing for DDR pipeline analysis.
    // generate_ticks: time spent issuing/waiting for MATH/PACK pattern generation.
    // ncrisc_blocked_wait_ticks: BRISC wall time blocked waiting for NCRISC completion.
    // compare_brisc_ticks: BRISC's local 1/4 compare loop time.
    // compare_wait_ticks: time after BRISC compare spent waiting for MATH/PACK/UNPACK compare helpers.
    // compare_total_ticks: compare wall time from start_compare to all helpers consumed.
    uint64_t generate_ticks;
    uint64_t ncrisc_blocked_wait_ticks;
    uint64_t compare_brisc_ticks;
    uint64_t compare_wait_ticks;
    uint64_t compare_total_ticks;

    // Granular profiler: active/idle counters reported by individual RISCVs.
    uint64_t ncrisc_idle_ticks;
    uint64_t ncrisc_write_active_ticks;
    uint64_t ncrisc_read_active_ticks;
    uint64_t ncrisc_diag_active_ticks;
    uint64_t math_generate_active_ticks;
    uint64_t pack_generate_active_ticks;
    uint64_t math_compare_active_ticks;
    uint64_t pack_compare_active_ticks;
    uint64_t unpack_compare_active_ticks;

    // BRISC wall time for the whole job, measured with device timestamp().
    uint64_t job_total_ticks;
};

// -------------------------
// Original runtime-arg job parameters
// -------------------------
struct DramTestParameters {
    uint32_t bank_id;
    uint32_t bank_offset_lo;
    uint32_t bank_offset_hi;
    uint32_t total_bytes;
    uint32_t chunk_bytes;
    uint32_t pattern_id;
    uint32_t seed;
    uint32_t pass_index;
    uint32_t repeat_index;
    uint32_t job_id;
    uint32_t result_l1_addr;
    uint32_t expect_l1_addr;
    uint32_t observe_l1_addr;
    uint32_t write_noc;
    uint32_t read_noc;
    uint32_t max_burst_len;
    uint32_t transfer_len_mode;
    uint32_t skip_writes;
    uint32_t skip_reads;
};

static inline uint64_t dram_test_bank_offset(const DramTestParameters& p) {
    return ((uint64_t)p.bank_offset_hi << 32) | (uint64_t)p.bank_offset_lo;
}

// -------------------------
// Persistent worker job item
// -------------------------
// This is the queueable equivalent of the runtime args above.
// Host fills this into a per-core queue; kernel consumes it.
struct DramWorkItem {
    uint32_t job_id;
    uint32_t bank_id;
    uint32_t bank_offset_lo;
    uint32_t bank_offset_hi;
    uint32_t total_bytes;
    uint32_t chunk_bytes;
    uint32_t pattern_id;
    uint32_t seed;
    uint32_t pass_index;
    uint32_t repeat_index;
    uint32_t write_noc;
    uint32_t read_noc;
    uint32_t max_burst_len;
    uint32_t transfer_len_mode;
    uint32_t skip_writes;
    uint32_t skip_reads;
};

// -------------------------
// Per-core queue control block
// -------------------------
// Host is producer, one worker core is consumer.
// head = next slot device consumes
// tail = next slot host fills
struct DramJobQueueCtrl {
    uint32_t magic;
    uint32_t head;
    uint32_t tail;
    uint32_t capacity;
    uint32_t stop_requested;
    uint32_t jobs_completed;
    uint32_t reserved0;
    uint32_t reserved1;
};

static constexpr uint32_t DRAM_JOB_QUEUE_MAGIC = 0x4A4F4244u;  // "DJOB"

static inline void dram_job_queue_ctrl_init(DramJobQueueCtrl& ctrl, uint32_t capacity = DRAM_JOB_QUEUE_CAPACITY) {
    ctrl.magic = DRAM_JOB_QUEUE_MAGIC;
    ctrl.head = 0u;
    ctrl.tail = 0u;
    ctrl.capacity = capacity;
    ctrl.stop_requested = 0u;
    ctrl.jobs_completed = 0u;
    ctrl.reserved0 = 0u;
    ctrl.reserved1 = 0u;
}

static inline uint32_t dram_job_queue_next_index(uint32_t idx, uint32_t capacity) { return (idx + 1u) % capacity; }

static inline bool dram_job_queue_is_empty(const volatile DramJobQueueCtrl* ctrl) { return ctrl->head == ctrl->tail; }

static inline bool dram_job_queue_is_full(const volatile DramJobQueueCtrl* ctrl) {
    return dram_job_queue_next_index(ctrl->tail, ctrl->capacity) == ctrl->head;
}

// -------------------------
// Per-core live progress / telemetry
// -------------------------
struct CoreProgressStatus {
    uint32_t magic;
    uint32_t state;
    uint32_t current_stage;
    uint32_t current_job_id;
    uint32_t jobs_completed;
    uint32_t heartbeat_tick;
    uint32_t reserved0;
    uint32_t reserved1;
};

static inline void dram_progress_status_init(CoreProgressStatus& status) {
    status.magic = DRAM_PROGRESS_MAGIC;
    status.state = DRAM_PROGRESS_STATE_IDLE;
    status.current_stage = DRAM_PROGRESS_STAGE_WAIT;
    status.current_job_id = 0u;
    status.jobs_completed = 0u;
    status.heartbeat_tick = 0u;
    status.reserved0 = 0u;
    status.reserved1 = 0u;
}

// -------------------------
// Optional packed per-slot layout helper
// -------------------------
// If you want one contiguous allocation per core in L1, this can help,
// but you are free to allocate queue ctrl / jobs / status / result ring
// separately on the host.
template <uint32_t QUEUE_CAPACITY = DRAM_JOB_QUEUE_CAPACITY>
struct DramPersistentWorkerLayout {
    DramJobQueueCtrl ctrl;
    CoreProgressStatus status;
    DramWorkItem jobs[QUEUE_CAPACITY];
    DramBaseResult results[QUEUE_CAPACITY];
};

// -------------------------
// Conversion helper
// -------------------------
static inline DramTestParameters dram_make_test_parameters_from_work_item(
    const DramWorkItem& job, uint32_t result_l1_addr, uint32_t expect_l1_addr, uint32_t observe_l1_addr) {
    DramTestParameters p{};
    p.bank_id = job.bank_id;
    p.bank_offset_lo = job.bank_offset_lo;
    p.bank_offset_hi = job.bank_offset_hi;
    p.total_bytes = job.total_bytes;
    p.chunk_bytes = job.chunk_bytes;
    p.pattern_id = job.pattern_id;
    p.seed = job.seed;
    p.pass_index = job.pass_index;
    p.repeat_index = job.repeat_index;
    p.result_l1_addr = result_l1_addr;
    p.expect_l1_addr = expect_l1_addr;
    p.observe_l1_addr = observe_l1_addr;
    p.write_noc = job.write_noc;
    p.read_noc = job.read_noc;
    p.max_burst_len = job.max_burst_len;
    p.transfer_len_mode = job.transfer_len_mode;
    p.skip_writes = job.skip_writes;
    p.skip_reads = job.skip_reads;
    return p;
}

#ifdef KERNEL_BUILD

static inline void dram_wait_for_job_or_stop(uint32_t wake_flag_l1_addr, volatile DramJobQueueCtrl* ctrl) {
    volatile tt_l1_ptr uint32_t* wake_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wake_flag_l1_addr);

    while (true) {
        invalidate_l1_cache();

        if (ctrl->stop_requested) {
            return;
        }

        if (!dram_job_queue_is_empty(ctrl)) {
            return;
        }

        if (*wake_flag != 0u) {
            *wake_flag = 0u;  // consume wake
            return;
        }
    }
}

static inline void dram_signal_job_done(uint32_t /*unused*/) {}

#else

static inline void dram_wait_for_job_or_stop(uint32_t /*wake_flag_l1_addr*/, volatile DramJobQueueCtrl* /*ctrl*/) {}

static inline void dram_signal_job_done(uint32_t /*unused*/) {}

#endif

#endif /* _DRAM_COMMON_H */
