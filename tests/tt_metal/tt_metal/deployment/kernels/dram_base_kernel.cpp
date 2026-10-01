// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "patterns/patterns.hpp"
#include "patterns/dram_pattern_fill.hpp"
#include "patterns/sync_mailbox.hpp"
#include "common_dram.hpp"
#include "dram_utils.hpp"
#include "timestamp.hpp"

// ============================================================
// Error injection configuration
// ============================================================
// 0 = disabled, 1 = enabled
#define INSERT_WRITE_ERRORS 0
#define INSERT_READ_ERRORS 0

// Number of injected failing words of each kind
#define WRITE_ERROR_COUNT 20
#define READ_ERROR_COUNT 20

// Spread errors out so they are rare.
// Example: every 1 MiB / 4B = every 262144 words.
#define WRITE_ERROR_STRIDE_WORDS 262144u
#define READ_ERROR_STRIDE_WORDS 393216u

// Start offsets (in words) so write/read errors do not overlap
#define WRITE_ERROR_START_WORD 1024u
#define READ_ERROR_START_WORD 2048u

// ============================================================
// Watchdog validation: intentional Tensix heartbeat stall
// ============================================================
// Runtime-controlled from host via DramJobQueueCtrl::reserved0.
// Host sets reserved0=1 when env DRAM_TEST_INJECT_TENSIX_HEARTBEAT_STALL=1.
// Keep the actual stall point fixed for the negative watchdog validation test.
static constexpr uint32_t DRAM_TEST_STALL_JOB_ID = 1u;
static constexpr uint32_t DRAM_TEST_STALL_AFTER_TRANSFERS = 10u;

static inline bool dram_inject_tensix_heartbeat_stall_enabled(volatile DramJobQueueCtrl* ctrl) {
    return ctrl != nullptr && ctrl->reserved0 != 0u;
}

static inline void dram_maybe_inject_tensix_heartbeat_stall(
    const DramTestParameters& p,
    volatile DramBaseResult* result,
    volatile DramJobQueueCtrl* ctrl,
    volatile CoreProgressStatus* status,
    volatile uint32_t* sync_mb) {
    if (dram_inject_tensix_heartbeat_stall_enabled(ctrl) && (p.job_id == DRAM_TEST_STALL_JOB_ID) &&
        (result->transfers >= DRAM_TEST_STALL_AFTER_TRANSFERS)) {
        status->current_stage = DRAM_PROGRESS_STAGE_VERIFY;
        status->current_job_id = p.job_id;
        sync_mb[MB_CURRENT_STAGE] = MB_STAGE_VERIFY;

        while (true) {
            // Intentional watchdog validation hang.
            // Do not update heartbeat here.
            // Expected host behavior:
            //   hb_delta  == 0
            //   jobs_delta == 0
            //   arc_delta > 0
            // => watchdog reason should become kernel_stall.
        }
    }
}

struct DramPendingDiagnostic {
    uint32_t valid;
    uint32_t fail_byte_offset;
    uint32_t expected;
    uint32_t observed;
};

static inline void dram_pending_diag_init(DramPendingDiagnostic* pending) {
    pending->valid = 0u;
    pending->fail_byte_offset = 0xFFFFFFFFu;
    pending->expected = 0u;
    pending->observed = 0u;
}

static inline void dram_queue_deferred_failure(
    volatile DramBaseResult* result,
    volatile uint32_t* sync_mb,
    DramPendingDiagnostic* pending,
    uint32_t fail_byte_offset,
    uint32_t expected,
    uint32_t observed) {
    if (pending->valid == 0u) {
        pending->valid = 1u;
        pending->fail_byte_offset = fail_byte_offset;
        pending->expected = expected;
        pending->observed = observed;
    }

    if (result->failures == 0u) {
        result->first_fail_addr = fail_byte_offset;
        result->first_expected = expected;
        result->first_observed = observed;
        result->failure_kind = DRAM_FAILURE_READ;
        result->readback_count = 0u;

        sync_mb[MB_FIRST_FAIL_ADDR] = result->first_fail_addr;
        sync_mb[MB_FIRST_EXPECTED] = result->first_expected;
        sync_mb[MB_FIRST_OBSERVED] = result->first_observed;
        sync_mb[MB_FAILURE_KIND] = result->failure_kind;
    }
}

static inline void dram_arm_deferred_diag_if_needed(
    volatile uint32_t* sync_mb, DramPendingDiagnostic* pending_ready, DramPendingDiagnostic* pending_in_flight) {
    if (pending_ready->valid == 0u) {
        sync_mb[MB_NCRISC_DIAG_REQUEST] = 0u;
        return;
    }

    *pending_in_flight = *pending_ready;
    dram_pending_diag_init(pending_ready);

    sync_mb[MB_NCRISC_DIAG_ADDR] = pending_in_flight->fail_byte_offset;
    sync_mb[MB_NCRISC_DIAG_EXPECTED] = pending_in_flight->expected;
    sync_mb[MB_NCRISC_DIAG_OBSERVED] = pending_in_flight->observed;
    sync_mb[MB_NCRISC_DIAG_READBACK0] = 0u;
    sync_mb[MB_NCRISC_DIAG_READBACK1] = 0u;
    sync_mb[MB_NCRISC_DIAG_READBACK2] = 0u;
    sync_mb[MB_NCRISC_DIAG_READBACK3] = 0u;
    sync_mb[MB_NCRISC_DIAG_READBACK4] = 0u;
    sync_mb[MB_NCRISC_DIAG_KIND] = DRAM_FAILURE_NONE;
    sync_mb[MB_NCRISC_DIAG_REQUEST] = 1u;
}

static inline void dram_consume_deferred_diag_result(
    volatile DramBaseResult* result, volatile uint32_t* sync_mb, DramPendingDiagnostic* pending) {
    if (pending->valid == 0u) {
        return;
    }

    const uint32_t classified_kind = sync_mb[MB_NCRISC_DIAG_KIND];
    if (classified_kind == DRAM_FAILURE_WRITE) {
        result->suspected_write_failures++;
    } else {
        result->suspected_read_failures++;
    }

    if (result->first_fail_addr == pending->fail_byte_offset) {
        result->failure_kind = classified_kind;
        result->readback_count = 5u;
        result->readback_data[0] = sync_mb[MB_NCRISC_DIAG_READBACK0];
        result->readback_data[1] = sync_mb[MB_NCRISC_DIAG_READBACK1];
        result->readback_data[2] = sync_mb[MB_NCRISC_DIAG_READBACK2];
        result->readback_data[3] = sync_mb[MB_NCRISC_DIAG_READBACK3];
        result->readback_data[4] = sync_mb[MB_NCRISC_DIAG_READBACK4];
        sync_mb[MB_FAILURE_KIND] = result->failure_kind;
    }

    dram_pending_diag_init(pending);
}

static inline void dram_consume_compare_helper_result(
    volatile DramBaseResult* result,
    volatile uint32_t* sync_mb,
    DramPendingDiagnostic* pending,
    uint32_t result_idx,
    uint32_t first_addr_idx,
    uint32_t first_expected_idx,
    uint32_t first_observed_idx) {
    const uint32_t helper_failures = sync_mb[result_idx];

    if (helper_failures == 0u) {
        return;
    }

    const uint32_t fail_byte_offset = sync_mb[first_addr_idx];
    const uint32_t expected = sync_mb[first_expected_idx];
    const uint32_t observed = sync_mb[first_observed_idx];

    if (fail_byte_offset != 0xFFFFFFFFu) {
        dram_queue_deferred_failure(result, sync_mb, pending, fail_byte_offset, expected, observed);
    }

    result->failures += helper_failures;
}

[[maybe_unused]] static inline void dprint_top_2kb_of_bank_if_in_range(
    const DramTestParameters& p,
    uint64_t bank_offset_base,
    uint32_t offset,
    uint32_t transfer_bytes,
    uint32_t* expect_words,
    uint32_t* observe_words) {
    if (p.pattern_id != DRAM_PATTERN_CHECKERBOARD) {
        return;
    }
    if (p.bank_id != 0u) {
        return;
    }
    if (p.pass_index != 0u) {
        return;
    }
    if (p.repeat_index != 0u) {
        return;
    }

    constexpr uint64_t dump_size_bytes = 2048ull;
    constexpr uint64_t target_end = (uint64_t)DRAM_TEST_MAX_BANK_BYTES;
    constexpr uint64_t target_start = target_end - dump_size_bytes;

    const uint64_t chunk_start = bank_offset_base + (uint64_t)offset;
    const uint64_t chunk_end = chunk_start + (uint64_t)transfer_bytes;

    if (chunk_end <= target_start || chunk_start >= target_end) {
        return;
    }

    const uint64_t dump_from = (chunk_start < target_start) ? target_start : chunk_start;
    const uint64_t dump_to = (chunk_end > target_end) ? target_end : chunk_end;

    const uint32_t first_word = (uint32_t)((dump_from - chunk_start) / sizeof(uint32_t));
    const uint32_t last_word = (uint32_t)((dump_to - chunk_start) / sizeof(uint32_t));

    // Disabled verbose top-2KB DRAM compare dump.
    // This produced excessive terminal spam like:
    // addr=0x... expected=0x... observed=0x...
    //
    // DPRINT << "=== DRAM top 2KB dump begin ===" << ENDL();
    // DPRINT << "bank=" << p.bank_id << " pass=" << p.pass_index << " repeat=" << p.repeat_index << " chunk_start=0x"
    //        << HEX() << chunk_start << " chunk_end=0x" << chunk_end << " dump_from=0x" << dump_from << " dump_to=0x"
    //        << dump_to << ENDL();
    //
    // for (uint32_t i = first_word; i < last_word; ++i) {
    //     const uint64_t abs_addr = chunk_start + (uint64_t)i * sizeof(uint32_t);
    //     DPRINT << "addr=0x" << HEX() << abs_addr << " expected=0x" << expect_words[i] << " observed=0x"
    //            << observe_words[i] << ENDL();
    // }
    //
    // DPRINT << "=== DRAM top 2KB dump end ===" << ENDL();
}

static inline void dram_status_heartbeat(volatile CoreProgressStatus* status, uint32_t stage) {
    status->current_stage = stage;
    status->heartbeat_tick++;
}

static inline bool dram_stop_requested(volatile DramJobQueueCtrl* ctrl) { return ctrl->stop_requested != 0u; }

static inline uint64_t dram_mb_read_u64(volatile uint32_t* mb, uint32_t lo_idx) {
    const uint64_t lo = mb[lo_idx];
    const uint64_t hi = mb[lo_idx + 1u];
    return (hi << 32) | lo;
}

static inline void dram_mb_reset_u64(volatile uint32_t* mb, uint32_t lo_idx) {
    mb[lo_idx] = 0u;
    mb[lo_idx + 1u] = 0u;
}

static inline DramTestParameters dram_make_params_from_work_item(
    const volatile DramWorkItem& job, uint32_t result_l1_addr, uint32_t expect_l1_addr, uint32_t observe_l1_addr) {
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
    p.job_id = job.job_id;
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

static inline void dram_reset_result(volatile DramBaseResult* result, const DramTestParameters& p) {
    result->job_id = 0xFFFFFFFFu;
    result->pattern_id = p.pattern_id;
    result->pass_index = p.pass_index;
    result->repeat_index = p.repeat_index;
    result->bank_id = p.bank_id;
    result->transfers = 0u;
    result->words_checked = 0u;
    result->failures = 0u;
    result->first_fail_addr = 0xFFFFFFFFu;
    result->first_expected = 0u;
    result->first_observed = 0u;
    result->failure_kind = DRAM_FAILURE_NONE;
    result->readback_count = 0u;
    result->suspected_write_failures = 0u;
    result->suspected_read_failures = 0u;
    result->prepare_ticks = 0u;
    result->write_ticks = 0u;
    result->read_ticks = 0u;
    result->generate_ticks = 0u;
    result->ncrisc_blocked_wait_ticks = 0u;
    result->compare_brisc_ticks = 0u;
    result->compare_wait_ticks = 0u;
    result->compare_total_ticks = 0u;
    result->ncrisc_idle_ticks = 0u;
    result->ncrisc_write_active_ticks = 0u;
    result->ncrisc_read_active_ticks = 0u;
    result->ncrisc_diag_active_ticks = 0u;
    result->math_generate_active_ticks = 0u;
    result->pack_generate_active_ticks = 0u;
    result->math_compare_active_ticks = 0u;
    result->pack_compare_active_ticks = 0u;
    result->unpack_compare_active_ticks = 0u;
    result->job_total_ticks = 0u;

    for (uint32_t i = 0; i < 5u; ++i) {
        result->readback_data[i] = 0u;
    }
}

struct DramChunkState {
    uint32_t offset = 0u;
    uint32_t transfer_bytes = 0u;
    uint32_t word_count = 0u;
    uint32_t base_word_index = 0u;
    uint32_t gen_slot = 0u;
    uint32_t gen_l1_addr = 0u;
    uint32_t obs_slot = 0u;
    uint32_t obs_l1_addr = 0u;
};

static inline uint32_t dram_get_gen_l1_addr(volatile uint32_t* sync_mb, uint32_t slot) {
    return (slot == DRAM_GEN_SLOT_PING) ? sync_mb[MB_GEN_PING_L1_ADDR] : sync_mb[MB_GEN_PONG_L1_ADDR];
}

static inline uint32_t dram_get_obs_l1_addr(volatile uint32_t* sync_mb, uint32_t slot) {
    return (slot == DRAM_OBS_SLOT_PING) ? sync_mb[MB_OBS_PING_L1_ADDR] : sync_mb[MB_OBS_PONG_L1_ADDR];
}

static inline DramChunkState dram_make_chunk_state(
    uint32_t offset, uint32_t transfer_bytes, uint32_t chunk_index, volatile uint32_t* sync_mb) {
    DramChunkState c{};

    c.offset = offset;
    c.transfer_bytes = transfer_bytes;
    c.word_count = transfer_bytes / sizeof(uint32_t);
    c.base_word_index = offset / sizeof(uint32_t);

    c.gen_slot = chunk_index & 1u;
    c.obs_slot = chunk_index & 1u;

    c.gen_l1_addr = dram_get_gen_l1_addr(sync_mb, c.gen_slot);
    c.obs_l1_addr = dram_get_obs_l1_addr(sync_mb, c.obs_slot);

    return c;
}

static inline void dram_start_generate_chunk(
    const DramChunkState& c, volatile uint32_t* sync_mb, uint32_t generate_tag) {
    sync_mb[MB_GEN_ACTIVE_SLOT] = c.gen_slot;
    sync_mb[MB_GEN_ACTIVE_L1_ADDR] = c.gen_l1_addr;

    sync_mb[MB_GENERATE_L1_ADDR] = c.gen_l1_addr;
    sync_mb[MB_GENERATE_WORD_COUNT] = c.word_count;
    sync_mb[MB_GENERATE_BASE_WORD_INDEX] = c.base_word_index;

    sync_mb[MB_GENERATE_MATH_DONE] = 0u;
    sync_mb[MB_GENERATE_PACK_DONE] = 0u;

    noc_async_write_barrier();
    sync_mb[MB_GENERATE_START] = generate_tag;
    noc_async_write_barrier();
}

static inline bool dram_wait_generate_chunk(
    volatile uint32_t* sync_mb,
    volatile DramJobQueueCtrl* ctrl,
    volatile CoreProgressStatus* status,
    uint32_t generate_tag) {
    while ((sync_mb[MB_GENERATE_MATH_DONE] != generate_tag) || (sync_mb[MB_GENERATE_PACK_DONE] != generate_tag)) {
        noc_async_read_barrier();

        if (dram_stop_requested(ctrl)) {
            sync_mb[MB_STOP] = 1u;
            sync_mb[MB_ERROR] = MB_ERROR_STOP_REQUESTED;
            status->current_stage = DRAM_PROGRESS_STAGE_DONE;
            return false;
        }
    }

    return true;
}

static inline void dram_start_ncrisc_io_chunk(
    const DramChunkState& c,
    volatile uint32_t* sync_mb,
    DramPendingDiagnostic* pending_diag_ready,
    DramPendingDiagnostic* pending_diag_in_flight,
    uint32_t ncrisc_tag) {
    sync_mb[MB_GEN_ACTIVE_SLOT] = c.gen_slot;
    sync_mb[MB_GEN_ACTIVE_L1_ADDR] = c.gen_l1_addr;

    sync_mb[MB_OBS_ACTIVE_SLOT] = c.obs_slot;
    sync_mb[MB_OBS_ACTIVE_L1_ADDR] = c.obs_l1_addr;

    sync_mb[MB_CURRENT_OFFSET_BYTES] = c.offset;
    sync_mb[MB_CURRENT_TRANSFER_BYTES] = c.transfer_bytes;
    sync_mb[MB_CURRENT_WORD_COUNT] = c.word_count;
    sync_mb[MB_CURRENT_BASE_WORD] = c.base_word_index;

    sync_mb[MB_NCRISC_ERROR] = MB_ERROR_NONE;
    dram_arm_deferred_diag_if_needed(sync_mb, pending_diag_ready, pending_diag_in_flight);

    noc_async_write_barrier();
    sync_mb[MB_NCRISC_START] = ncrisc_tag;
    noc_async_write_barrier();
}

static inline void dram_start_ncrisc_diag_only(
    volatile uint32_t* sync_mb,
    DramPendingDiagnostic* pending_diag_ready,
    DramPendingDiagnostic* pending_diag_in_flight,
    uint32_t ncrisc_tag) {
    sync_mb[MB_CURRENT_OFFSET_BYTES] = 0u;
    sync_mb[MB_CURRENT_TRANSFER_BYTES] = 0u;
    sync_mb[MB_CURRENT_WORD_COUNT] = 0u;
    sync_mb[MB_CURRENT_BASE_WORD] = 0u;
    sync_mb[MB_NCRISC_ERROR] = MB_ERROR_NONE;
    dram_arm_deferred_diag_if_needed(sync_mb, pending_diag_ready, pending_diag_in_flight);

    noc_async_write_barrier();
    sync_mb[MB_NCRISC_START] = ncrisc_tag;
    noc_async_write_barrier();
}

static inline bool dram_wait_ncrisc_io_chunk(
    volatile uint32_t* sync_mb,
    volatile DramJobQueueCtrl* ctrl,
    volatile CoreProgressStatus* status,
    uint32_t ncrisc_tag) {
    while (sync_mb[MB_NCRISC_DONE] != ncrisc_tag) {
        noc_async_read_barrier();

        if (dram_stop_requested(ctrl)) {
            sync_mb[MB_STOP] = 1u;
            sync_mb[MB_ERROR] = MB_ERROR_STOP_REQUESTED;
            status->current_stage = DRAM_PROGRESS_STAGE_DONE;
            return false;
        }
    }

    return true;
}

static inline void dram_start_compare_chunk(const DramChunkState& c, volatile uint32_t* sync_mb, uint32_t compare_tag) {
    sync_mb[MB_COMPARE_SOURCE_L1_ADDR] = c.gen_l1_addr;
    sync_mb[MB_COMPARE_OBSERVED_L1_ADDR] = c.obs_l1_addr;
    sync_mb[MB_COMPARE_WORD_COUNT] = c.word_count;
    sync_mb[MB_COMPARE_BASE_BYTE_OFFSET] = c.offset;
    sync_mb[MB_COMPARE_MATH_RESULT] = 0u;
    sync_mb[MB_COMPARE_PACK_RESULT] = 0u;
    sync_mb[MB_COMPARE_UNPACK_RESULT] = 0u;
    // NCRISC compares the first quarter immediately after its read phase.
    // Do not reset the NCRISC compare result here: BRISC consumes it right after
    // waiting for MB_NCRISC_DONE, before it can feed the next NCRISC command.
    sync_mb[MB_COMPARE_MATH_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_MATH_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_MATH_FIRST_OBSERVED] = 0u;
    sync_mb[MB_COMPARE_PACK_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_PACK_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_PACK_FIRST_OBSERVED] = 0u;
    sync_mb[MB_COMPARE_UNPACK_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_UNPACK_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_UNPACK_FIRST_OBSERVED] = 0u;

    noc_async_write_barrier();
    sync_mb[MB_COMPARE_MATH_DONE] = 0u;
    sync_mb[MB_COMPARE_PACK_DONE] = 0u;
    sync_mb[MB_COMPARE_UNPACK_DONE] = 0u;
    sync_mb[MB_COMPARE_START] = compare_tag;
    noc_async_write_barrier();
}

static inline bool dram_finish_compare_chunk(
    const DramTestParameters& p,
    const DramChunkState& c,
    volatile DramBaseResult* result,
    volatile uint32_t* sync_mb,
    volatile DramJobQueueCtrl* ctrl,
    volatile CoreProgressStatus* status,
    DramPendingDiagnostic* pending_diag,
    uint32_t compare_tag,
    uint64_t* total_compare_brisc_ticks,
    uint64_t* total_compare_wait_ticks,
    uint64_t* total_compare_total_ticks) {
    const uint64_t compare_t0 = timestamp();
    const uint32_t failures_before_compare = result->failures;

    // BRISC no longer compares locally. NCRISC already compared the first quarter
    // immediately after its read phase; MATH/PACK/UNPACK compare the remaining
    // three quarters. BRISC only waits for helper done tags and aggregates results.
    const uint64_t compare_brisc_done_t = compare_t0;

    while ((sync_mb[MB_COMPARE_MATH_DONE] != compare_tag) || (sync_mb[MB_COMPARE_PACK_DONE] != compare_tag) ||
           (sync_mb[MB_COMPARE_UNPACK_DONE] != compare_tag)) {
        noc_async_read_barrier();
        if (dram_stop_requested(ctrl)) {
            sync_mb[MB_STOP] = 1u;
            sync_mb[MB_ERROR] = MB_ERROR_STOP_REQUESTED;
            status->current_stage = DRAM_PROGRESS_STAGE_DONE;
            return false;
        }
    }

    result->words_checked += c.word_count;

    dram_consume_compare_helper_result(
        result,
        sync_mb,
        pending_diag,
        MB_COMPARE_MATH_RESULT,
        MB_COMPARE_MATH_FIRST_ADDR,
        MB_COMPARE_MATH_FIRST_EXPECTED,
        MB_COMPARE_MATH_FIRST_OBSERVED);
    dram_consume_compare_helper_result(
        result,
        sync_mb,
        pending_diag,
        MB_COMPARE_PACK_RESULT,
        MB_COMPARE_PACK_FIRST_ADDR,
        MB_COMPARE_PACK_FIRST_EXPECTED,
        MB_COMPARE_PACK_FIRST_OBSERVED);
    dram_consume_compare_helper_result(
        result,
        sync_mb,
        pending_diag,
        MB_COMPARE_UNPACK_RESULT,
        MB_COMPARE_UNPACK_FIRST_ADDR,
        MB_COMPARE_UNPACK_FIRST_EXPECTED,
        MB_COMPARE_UNPACK_FIRST_OBSERVED);

    if (result->failures > failures_before_compare) {
        // Mismatch is reported through DramBaseResult and host log_info.
    }

    const uint64_t compare_done_t = timestamp();
    *total_compare_brisc_ticks += (compare_brisc_done_t - compare_t0);
    *total_compare_wait_ticks += (compare_done_t - compare_brisc_done_t);
    *total_compare_total_ticks += (compare_done_t - compare_t0);
    return true;
}

static inline bool run_one_dram_job(
    const DramTestParameters& p,
    volatile DramBaseResult* result,
    uint32_t* expect_words,
    uint32_t* observe_words,
    volatile CoreProgressStatus* status,
    volatile DramJobQueueCtrl* ctrl,
    volatile uint32_t* sync_mb) {
    (void)expect_words;
    (void)observe_words;

    const uint64_t job_total_t0 = timestamp();
    const uint64_t bank_offset_base = dram_test_bank_offset(p);

    dram_reset_result(result, p);

    sync_mb[MB_STOP] = 0u;
    sync_mb[MB_ERROR] = MB_ERROR_NONE;

    sync_mb[MB_JOB_ID] = p.job_id;
    sync_mb[MB_BANK_ID] = p.bank_id;
    sync_mb[MB_BANK_OFFSET_LO] = p.bank_offset_lo;
    sync_mb[MB_BANK_OFFSET_HI] = p.bank_offset_hi;
    sync_mb[MB_TOTAL_BYTES] = p.total_bytes;
    sync_mb[MB_CHUNK_BYTES] = p.chunk_bytes;
    sync_mb[MB_PATTERN_ID_GLOBAL] = p.pattern_id;
    sync_mb[MB_SEED_GLOBAL] = p.seed;
    sync_mb[MB_PASS_INDEX_GLOBAL] = p.pass_index;
    sync_mb[MB_REPEAT_INDEX_GLOBAL] = p.repeat_index;

    sync_mb[MB_RESULT_L1_ADDR] = p.result_l1_addr;
    sync_mb[MB_EXPECT_L1_ADDR] = p.expect_l1_addr;
    sync_mb[MB_OBSERVE_L1_ADDR] = p.observe_l1_addr;

    if (sync_mb[MB_GEN_PING_L1_ADDR] == 0u) {
        sync_mb[MB_GEN_PING_L1_ADDR] = p.expect_l1_addr;
    }

    if (sync_mb[MB_GEN_PONG_L1_ADDR] == 0u) {
        sync_mb[MB_GEN_PONG_L1_ADDR] = p.expect_l1_addr;
    }

    sync_mb[MB_GEN_ACTIVE_SLOT] = DRAM_GEN_SLOT_PING;
    sync_mb[MB_GEN_ACTIVE_L1_ADDR] = sync_mb[MB_GEN_PING_L1_ADDR];

    if (sync_mb[MB_OBS_PING_L1_ADDR] == 0u) {
        sync_mb[MB_OBS_PING_L1_ADDR] = p.observe_l1_addr;
    }

    if (sync_mb[MB_OBS_PONG_L1_ADDR] == 0u) {
        sync_mb[MB_OBS_PONG_L1_ADDR] = p.observe_l1_addr;
    }

    sync_mb[MB_OBS_ACTIVE_SLOT] = DRAM_OBS_SLOT_PING;
    sync_mb[MB_OBS_ACTIVE_L1_ADDR] = sync_mb[MB_OBS_PING_L1_ADDR];

    sync_mb[MB_WRITE_NOC] = p.write_noc;
    sync_mb[MB_READ_NOC] = p.read_noc;
    sync_mb[MB_MAX_BURST_LEN] = p.max_burst_len;
    sync_mb[MB_TRANSFER_LEN_MODE] = p.transfer_len_mode;
    sync_mb[MB_SKIP_WRITES] = p.skip_writes;
    sync_mb[MB_SKIP_READS] = p.skip_reads;

    sync_mb[MB_CURRENT_STAGE] = MB_STAGE_JOB_START;
    sync_mb[MB_CURRENT_CHUNK] = 0u;
    sync_mb[MB_TOTAL_CHUNKS] = 0u;
    sync_mb[MB_TRANSFERS] = 0u;
    sync_mb[MB_WORDS_CHECKED] = 0u;
    sync_mb[MB_FAILURES] = 0u;
    sync_mb[MB_FIRST_FAIL_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_FIRST_EXPECTED] = 0u;
    sync_mb[MB_FIRST_OBSERVED] = 0u;
    sync_mb[MB_FAILURE_KIND] = DRAM_FAILURE_NONE;
    sync_mb[MB_SUSPECTED_WRITE_FAILURES] = 0u;
    sync_mb[MB_SUSPECTED_READ_FAILURES] = 0u;
    sync_mb[MB_NCRISC_DIAG_REQUEST] = 0u;
    sync_mb[MB_NCRISC_DIAG_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_NCRISC_DIAG_EXPECTED] = 0u;
    sync_mb[MB_NCRISC_DIAG_OBSERVED] = 0u;
    sync_mb[MB_NCRISC_DIAG_KIND] = DRAM_FAILURE_NONE;

    dram_mb_reset_u64(sync_mb, MB_PROF_NCRISC_IDLE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_NCRISC_WRITE_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_NCRISC_READ_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_NCRISC_DIAG_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_MATH_GEN_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_PACK_GEN_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_MATH_CMP_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_PACK_CMP_ACTIVE_LO);
    dram_mb_reset_u64(sync_mb, MB_PROF_UNPACK_CMP_ACTIVE_LO);

    //    sync_mb[MB_NCRISC_START] = 0u;
    //    sync_mb[MB_NCRISC_DONE] = 0u;
    sync_mb[MB_NCRISC_ERROR] = MB_ERROR_NONE;
    sync_mb[MB_NCRISC_ACTIVE_OFFSET_BYTES] = 0u;
    sync_mb[MB_NCRISC_ACTIVE_TRANSFER_BYTES] = 0u;

    //    sync_mb[MB_GENERATE_START] = 0u;
    //    sync_mb[MB_GENERATE_DONE] = 0u;
    sync_mb[MB_GENERATE_L1_ADDR] = 0u;
    sync_mb[MB_GENERATE_WORD_COUNT] = 0u;
    sync_mb[MB_GENERATE_BASE_WORD_INDEX] = 0u;
    //    sync_mb[MB_GENERATE_MATH_DONE] = 0u;
    //    sync_mb[MB_GENERATE_PACK_DONE] = 0u;

    //    sync_mb[MB_COMPARE_START] = 0u;
    //    sync_mb[MB_COMPARE_DONE] = 0u;
    sync_mb[MB_COMPARE_SOURCE_L1_ADDR] = 0u;
    sync_mb[MB_COMPARE_OBSERVED_L1_ADDR] = 0u;
    sync_mb[MB_COMPARE_WORD_COUNT] = 0u;
    sync_mb[MB_COMPARE_BASE_BYTE_OFFSET] = 0u;

    //    sync_mb[MB_COMPARE_MATH_DONE] = 0u;
    sync_mb[MB_COMPARE_MATH_RESULT] = 0u;
    sync_mb[MB_COMPARE_MATH_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_MATH_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_MATH_FIRST_OBSERVED] = 0u;

    //    sync_mb[MB_COMPARE_PACK_DONE] = 0u;
    sync_mb[MB_COMPARE_PACK_RESULT] = 0u;
    sync_mb[MB_COMPARE_PACK_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_PACK_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_PACK_FIRST_OBSERVED] = 0u;

    //    sync_mb[MB_COMPARE_UNPACK_DONE] = 0u;
    sync_mb[MB_COMPARE_UNPACK_RESULT] = 0u;
    sync_mb[MB_COMPARE_UNPACK_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_UNPACK_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_UNPACK_FIRST_OBSERVED] = 0u;

    sync_mb[MB_COMPARE_NCRISC_DONE] = 0u;
    sync_mb[MB_COMPARE_NCRISC_RESULT] = 0u;
    sync_mb[MB_COMPARE_NCRISC_FIRST_ADDR] = 0xFFFFFFFFu;
    sync_mb[MB_COMPARE_NCRISC_FIRST_EXPECTED] = 0u;
    sync_mb[MB_COMPARE_NCRISC_FIRST_OBSERVED] = 0u;

    if (dram_stop_requested(ctrl)) {
        sync_mb[MB_STOP] = 1u;
        sync_mb[MB_ERROR] = MB_ERROR_STOP_REQUESTED;
        status->current_stage = DRAM_PROGRESS_STAGE_DONE;
        return false;
    }

    if ((p.chunk_bytes == 0u) || ((p.chunk_bytes & 0x3u) != 0u) || ((p.total_bytes & 0x3u) != 0u)) {
        result->failures = 1u;
        result->first_fail_addr = 0u;
        result->first_expected = 0u;
        result->first_observed = p.chunk_bytes;

        sync_mb[MB_ERROR] = MB_ERROR_BAD_CONFIG;
        sync_mb[MB_FAILURES] = result->failures;
        sync_mb[MB_FIRST_FAIL_ADDR] = result->first_fail_addr;
        sync_mb[MB_FIRST_EXPECTED] = result->first_expected;
        sync_mb[MB_FIRST_OBSERVED] = result->first_observed;

        result->job_total_ticks = timestamp() - job_total_t0;
        noc_async_write_barrier();
        result->job_id = p.job_id;
        noc_async_write_barrier();
        return true;
    }

    uint64_t total_prepare_ticks = 0u;
    uint64_t total_write_ticks = 0u;
    uint64_t total_read_ticks = 0u;
    uint64_t total_generate_ticks = 0u;
    uint64_t total_ncrisc_blocked_wait_ticks = 0u;
    uint64_t total_compare_brisc_ticks = 0u;
    uint64_t total_compare_wait_ticks = 0u;
    uint64_t total_compare_total_ticks = 0u;

    uint32_t rng_state = p.seed ^ p.pass_index;
    if (rng_state == 0u) {
        rng_state = 1u;
    }

    uint32_t generate_request_tag = sync_mb[MB_GENERATE_START];
    uint32_t ncrisc_request_tag = sync_mb[MB_NCRISC_START];
    uint32_t compare_request_tag = sync_mb[MB_COMPARE_START];

    const uint32_t reuse_period_words = dram_pattern_reuse_period_words(p.pattern_id);
    const bool reuse_generated_pattern = (reuse_period_words != 0u);
    bool generated_slot_valid[2] = {false, false};
    uint32_t generated_slot_word_count[2] = {0u, 0u};
    uint32_t generated_slot_phase[2] = {0u, 0u};

    uint32_t next_offset = 0u;
    uint32_t chunk_index = 0u;

    bool have_current = false;
    DramChunkState current_chunk{};

    bool have_next = false;
    DramChunkState next_chunk{};

    bool current_io_in_flight = false;
    uint32_t current_io_tag = 0u;
    uint64_t current_io_t0 = 0u;

    DramPendingDiagnostic pending_diag_ready{};
    DramPendingDiagnostic pending_diag_in_flight{};
    dram_pending_diag_init(&pending_diag_ready);
    dram_pending_diag_init(&pending_diag_in_flight);

    if (next_offset < p.total_bytes) {
        const uint32_t remaining_bytes = p.total_bytes - next_offset;
        const uint32_t transfer_bytes = dram_choose_transfer_len(p, remaining_bytes, rng_state);

        current_chunk = dram_make_chunk_state(next_offset, transfer_bytes, chunk_index, sync_mb);

        dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_PREPARE);
        sync_mb[MB_CURRENT_STAGE] = MB_STAGE_PREPARE;
        sync_mb[MB_CURRENT_CHUNK] = chunk_index;

        const uint64_t prep_t0 = timestamp();

        const uint32_t gen_slot = current_chunk.gen_slot & 1u;
        const uint32_t chunk_phase =
            (reuse_period_words > 1u) ? (current_chunk.base_word_index % reuse_period_words) : 0u;
        const bool reusable_slot = reuse_generated_pattern && generated_slot_valid[gen_slot] &&
                                   (generated_slot_word_count[gen_slot] >= current_chunk.word_count) &&
                                   ((reuse_period_words == 1u) || (generated_slot_phase[gen_slot] == chunk_phase));
        const bool need_generate = !reusable_slot;

        if (need_generate) {
            ++generate_request_tag;
            if (generate_request_tag == 0u) {
                generate_request_tag = 1u;
            }

            const uint64_t gen_t0 = timestamp();
            dram_start_generate_chunk(current_chunk, sync_mb, generate_request_tag);

            if (!dram_wait_generate_chunk(sync_mb, ctrl, status, generate_request_tag)) {
                return false;
            }
            const uint64_t gen_t1 = timestamp();
            total_generate_ticks += (gen_t1 - gen_t0);

            generated_slot_valid[gen_slot] = true;
            generated_slot_word_count[gen_slot] = current_chunk.word_count;
            generated_slot_phase[gen_slot] = chunk_phase;
        }

        const uint64_t prep_t1 = timestamp();
        total_prepare_ticks += (prep_t1 - prep_t0);

        next_offset += transfer_bytes;
        chunk_index++;
        have_current = true;
    }

    while (have_current) {
        if (dram_stop_requested(ctrl)) {
            sync_mb[MB_STOP] = 1u;
            sync_mb[MB_ERROR] = MB_ERROR_STOP_REQUESTED;
            status->current_stage = DRAM_PROGRESS_STAGE_DONE;
            return false;
        }

        sync_mb[MB_CURRENT_CHUNK] = result->transfers;
        sync_mb[MB_CURRENT_OFFSET_BYTES] = current_chunk.offset;
        sync_mb[MB_CURRENT_TRANSFER_BYTES] = current_chunk.transfer_bytes;
        sync_mb[MB_CURRENT_WORD_COUNT] = current_chunk.word_count;
        sync_mb[MB_CURRENT_BASE_WORD] = current_chunk.base_word_index;

        uint64_t io_t0 = 0u;
        uint32_t active_ncrisc_tag = 0u;
        bool active_io = false;

        if (current_io_in_flight) {
            io_t0 = current_io_t0;
            active_ncrisc_tag = current_io_tag;
            active_io = true;
            current_io_in_flight = false;
        } else {
            io_t0 = timestamp();
            if (!p.skip_writes) {
                dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_WRITE);
                sync_mb[MB_CURRENT_STAGE] = MB_STAGE_WRITE;
            } else if (!p.skip_reads) {
                dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_READ);
                sync_mb[MB_CURRENT_STAGE] = MB_STAGE_READ;
            }
            if (!p.skip_writes || !p.skip_reads) {
                ++ncrisc_request_tag;
                if (ncrisc_request_tag == 0u) {
                    ncrisc_request_tag = 1u;
                }
                active_ncrisc_tag = ncrisc_request_tag;
                dram_start_ncrisc_io_chunk(
                    current_chunk, sync_mb, &pending_diag_ready, &pending_diag_in_flight, active_ncrisc_tag);
                active_io = true;
            }
        }

        have_next = false;

        if (next_offset < p.total_bytes) {
            const uint32_t remaining_bytes = p.total_bytes - next_offset;
            const uint32_t transfer_bytes = dram_choose_transfer_len(p, remaining_bytes, rng_state);
            next_chunk = dram_make_chunk_state(next_offset, transfer_bytes, chunk_index, sync_mb);

            dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_PREPARE);
            sync_mb[MB_CURRENT_STAGE] = MB_STAGE_PREPARE;
            const uint64_t prep_t0 = timestamp();

            const uint32_t gen_slot = next_chunk.gen_slot & 1u;
            const uint32_t chunk_phase =
                (reuse_period_words > 1u) ? (next_chunk.base_word_index % reuse_period_words) : 0u;
            const bool reusable_slot = reuse_generated_pattern && generated_slot_valid[gen_slot] &&
                                       (generated_slot_word_count[gen_slot] >= next_chunk.word_count) &&
                                       ((reuse_period_words == 1u) || (generated_slot_phase[gen_slot] == chunk_phase));
            const bool need_generate = !reusable_slot;

            if (need_generate) {
                ++generate_request_tag;
                if (generate_request_tag == 0u) {
                    generate_request_tag = 1u;
                }
                const uint64_t gen_t0 = timestamp();
                dram_start_generate_chunk(next_chunk, sync_mb, generate_request_tag);
                if (!dram_wait_generate_chunk(sync_mb, ctrl, status, generate_request_tag)) {
                    return false;
                }
                const uint64_t gen_t1 = timestamp();
                total_generate_ticks += (gen_t1 - gen_t0);
                generated_slot_valid[gen_slot] = true;
                generated_slot_word_count[gen_slot] = next_chunk.word_count;
                generated_slot_phase[gen_slot] = chunk_phase;
            }

            const uint64_t prep_t1 = timestamp();
            total_prepare_ticks += (prep_t1 - prep_t0);
            next_offset += transfer_bytes;
            chunk_index++;
            have_next = true;
        }

        uint64_t io_t1 = io_t0;
        if (active_io) {
            const uint64_t ncrisc_wait_t0 = timestamp();
            if (!dram_wait_ncrisc_io_chunk(sync_mb, ctrl, status, active_ncrisc_tag)) {
                return false;
            }
            const uint64_t ncrisc_wait_t1 = timestamp();
            total_ncrisc_blocked_wait_ticks += (ncrisc_wait_t1 - ncrisc_wait_t0);
            io_t1 = ncrisc_wait_t1;

            if (sync_mb[MB_NCRISC_ERROR] != MB_ERROR_NONE) {
                result->failures = 1u;
                result->first_fail_addr = current_chunk.offset;
                result->first_expected = 0u;
                result->first_observed = sync_mb[MB_NCRISC_ERROR];
                result->failure_kind = DRAM_FAILURE_READ;
                sync_mb[MB_FAILURES] = result->failures;
                sync_mb[MB_FIRST_FAIL_ADDR] = result->first_fail_addr;
                sync_mb[MB_FIRST_EXPECTED] = result->first_expected;
                sync_mb[MB_FIRST_OBSERVED] = result->first_observed;
                sync_mb[MB_FAILURE_KIND] = result->failure_kind;
                sync_mb[MB_ERROR] = sync_mb[MB_NCRISC_ERROR];

                result->transfers++;
                status->heartbeat_tick++;
                sync_mb[MB_TRANSFERS] = result->transfers;
                sync_mb[MB_WORDS_CHECKED] = result->words_checked;
                sync_mb[MB_FAILURES] = result->failures;

                dram_maybe_inject_tensix_heartbeat_stall(p, result, ctrl, status, sync_mb);
                current_chunk = next_chunk;
                have_current = have_next;
                continue;
            }
            dram_consume_deferred_diag_result(result, sync_mb, &pending_diag_in_flight);

            if (!p.skip_reads) {
                dram_consume_compare_helper_result(
                    result,
                    sync_mb,
                    &pending_diag_ready,
                    MB_COMPARE_NCRISC_RESULT,
                    MB_COMPARE_NCRISC_FIRST_ADDR,
                    MB_COMPARE_NCRISC_FIRST_EXPECTED,
                    MB_COMPARE_NCRISC_FIRST_OBSERVED);
            }

            if (!p.skip_writes) {
                total_write_ticks += (io_t1 - io_t0);
            }
            if (!p.skip_reads) {
                total_read_ticks += (io_t1 - io_t0);
            }
        }

        // Feed NCRISC as early as possible. At this point chunk N I/O is complete,
        // and chunk N+1 has already been generated into the other ping/pong slot.
        // Start N+1 before launching compare for N, so NCRISC does not sit idle
        // during BRISC compare setup / helper dispatch.
        if (have_next && (!p.skip_writes || !p.skip_reads) && !current_io_in_flight) {
            ++ncrisc_request_tag;
            if (ncrisc_request_tag == 0u) {
                ncrisc_request_tag = 1u;
            }
            current_io_tag = ncrisc_request_tag;
            current_io_t0 = timestamp();
            dram_start_ncrisc_io_chunk(
                next_chunk, sync_mb, &pending_diag_ready, &pending_diag_in_flight, current_io_tag);
            current_io_in_flight = true;
        }

        if (!p.skip_reads) {
            uint32_t* active_expect_words = reinterpret_cast<uint32_t*>(current_chunk.gen_l1_addr);
            uint32_t* active_observe_words = reinterpret_cast<uint32_t*>(current_chunk.obs_l1_addr);
            dprint_top_2kb_of_bank_if_in_range(
                p,
                bank_offset_base,
                current_chunk.offset,
                current_chunk.transfer_bytes,
                active_expect_words,
                active_observe_words);

            dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_VERIFY);
            sync_mb[MB_CURRENT_STAGE] = MB_STAGE_VERIFY;
            ++compare_request_tag;
            if (compare_request_tag == 0u) {
                compare_request_tag = 1u;
            }
            dram_start_compare_chunk(current_chunk, sync_mb, compare_request_tag);

            if (!dram_finish_compare_chunk(
                    p,
                    current_chunk,
                    result,
                    sync_mb,
                    ctrl,
                    status,
                    &pending_diag_ready,
                    compare_request_tag,
                    &total_compare_brisc_ticks,
                    &total_compare_wait_ticks,
                    &total_compare_total_ticks)) {
                return false;
            }
        }

        result->transfers++;
        status->heartbeat_tick++;

        sync_mb[MB_TRANSFERS] = result->transfers;
        sync_mb[MB_WORDS_CHECKED] = result->words_checked;
        sync_mb[MB_FAILURES] = result->failures;

        dram_maybe_inject_tensix_heartbeat_stall(p, result, ctrl, status, sync_mb);

        current_chunk = next_chunk;
        have_current = have_next;
    }

    if (pending_diag_ready.valid != 0u) {
        ++ncrisc_request_tag;
        if (ncrisc_request_tag == 0u) {
            ncrisc_request_tag = 1u;
        }
        const uint64_t ncrisc_wait_t0 = timestamp();
        dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_REREAD);
        sync_mb[MB_CURRENT_STAGE] = MB_STAGE_REREAD;
        dram_start_ncrisc_diag_only(sync_mb, &pending_diag_ready, &pending_diag_in_flight, ncrisc_request_tag);
        if (!dram_wait_ncrisc_io_chunk(sync_mb, ctrl, status, ncrisc_request_tag)) {
            return false;
        }
        const uint64_t ncrisc_wait_t1 = timestamp();
        total_ncrisc_blocked_wait_ticks += (ncrisc_wait_t1 - ncrisc_wait_t0);
        dram_consume_deferred_diag_result(result, sync_mb, &pending_diag_in_flight);
    }

    result->prepare_ticks = total_prepare_ticks;
    result->write_ticks = total_write_ticks;
    result->read_ticks = total_read_ticks;
    result->generate_ticks = total_generate_ticks;
    result->ncrisc_blocked_wait_ticks = total_ncrisc_blocked_wait_ticks;
    result->compare_brisc_ticks = total_compare_brisc_ticks;
    result->compare_wait_ticks = total_compare_wait_ticks;
    result->compare_total_ticks = total_compare_total_ticks;
    result->ncrisc_idle_ticks = dram_mb_read_u64(sync_mb, MB_PROF_NCRISC_IDLE_LO);
    result->ncrisc_write_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_NCRISC_WRITE_ACTIVE_LO);
    result->ncrisc_read_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_NCRISC_READ_ACTIVE_LO);
    result->ncrisc_diag_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_NCRISC_DIAG_ACTIVE_LO);
    result->math_generate_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_MATH_GEN_ACTIVE_LO);
    result->pack_generate_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_PACK_GEN_ACTIVE_LO);
    result->math_compare_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_MATH_CMP_ACTIVE_LO);
    result->pack_compare_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_PACK_CMP_ACTIVE_LO);
    result->unpack_compare_active_ticks = dram_mb_read_u64(sync_mb, MB_PROF_UNPACK_CMP_ACTIVE_LO);

    sync_mb[MB_CURRENT_STAGE] = MB_STAGE_JOB_DONE;
    sync_mb[MB_TRANSFERS] = result->transfers;
    sync_mb[MB_WORDS_CHECKED] = result->words_checked;
    sync_mb[MB_FAILURES] = result->failures;
    sync_mb[MB_SUSPECTED_WRITE_FAILURES] = result->suspected_write_failures;
    sync_mb[MB_SUSPECTED_READ_FAILURES] = result->suspected_read_failures;

    result->job_total_ticks = timestamp() - job_total_t0;
    noc_async_write_barrier();
    result->job_id = p.job_id;
    noc_async_write_barrier();
    return true;
}

void kernel_main() {
    const uint32_t queue_ctrl_l1_addr = get_arg_val<uint32_t>(0);
    const uint32_t queue_jobs_l1_addr = get_arg_val<uint32_t>(1);
    const uint32_t status_l1_addr = get_arg_val<uint32_t>(2);
    const uint32_t result_ring_l1_addr = get_arg_val<uint32_t>(3);
    const uint32_t expect_l1_addr = get_arg_val<uint32_t>(4);
    const uint32_t observe_l1_addr = get_arg_val<uint32_t>(5);
    const uint32_t queue_capacity = get_arg_val<uint32_t>(6);
    const uint32_t wake_flag_l1_addr = get_arg_val<uint32_t>(7);
    const uint32_t sync_mailbox_l1_addr = get_arg_val<uint32_t>(8);

    (void)wake_flag_l1_addr;

    volatile DramJobQueueCtrl* ctrl = reinterpret_cast<volatile DramJobQueueCtrl*>(queue_ctrl_l1_addr);
    volatile DramWorkItem* jobs = reinterpret_cast<volatile DramWorkItem*>(queue_jobs_l1_addr);
    volatile CoreProgressStatus* status = reinterpret_cast<volatile CoreProgressStatus*>(status_l1_addr);
    volatile DramBaseResult* result_ring = reinterpret_cast<volatile DramBaseResult*>(result_ring_l1_addr);
    volatile uint32_t* sync_mb = reinterpret_cast<volatile uint32_t*>(sync_mailbox_l1_addr);

    uint32_t* expect_words = reinterpret_cast<uint32_t*>(expect_l1_addr);
    uint32_t* observe_words = reinterpret_cast<uint32_t*>(observe_l1_addr);

    status->magic = DRAM_PROGRESS_MAGIC;
    status->state = DRAM_PROGRESS_STATE_IDLE;
    status->current_stage = DRAM_PROGRESS_STAGE_WAIT;
    status->current_job_id = 0u;
    status->jobs_completed = 0u;
    status->heartbeat_tick = 1u;

    if (ctrl->magic != DRAM_JOB_QUEUE_MAGIC || ctrl->capacity != queue_capacity) {
        status->state = DRAM_PROGRESS_STATE_ERROR;
        if (sync_mb[MB_MAGIC] == DRAM_SYNC_MAILBOX_MAGIC) {
            sync_mb[MB_ERROR] = MB_ERROR_BAD_CONFIG;
            sync_mb[MB_STOP] = 1u;
        }
        return;
    }

    if (sync_mb[MB_MAGIC] != DRAM_SYNC_MAILBOX_MAGIC) {
        status->state = DRAM_PROGRESS_STATE_ERROR;
        return;
    }

    /*
     * IMPORTANT:
     * Do not reset MB_*_START or MB_*_DONE here after helper kernels have started,
     * unless this is guaranteed to happen before NCRISC/compute observe the mailbox.
     *
     * Host zero_mailbox initialization is enough.
     */

    sync_mb[MB_STOP] = 0u;
    sync_mb[MB_ERROR] = MB_ERROR_NONE;
    sync_mb[MB_CURRENT_STAGE] = MB_STAGE_IDLE;

    while (true) {
        noc_async_read_barrier();

        if (ctrl->stop_requested) {
            break;
        }

        while (true) {
            noc_async_read_barrier();

            const uint32_t head = ctrl->head;
            const uint32_t tail = ctrl->tail;

            if (head == tail) {
                break;
            }

            const uint32_t slot = head % queue_capacity;
            const volatile DramWorkItem& job = jobs[slot];

            status->state = DRAM_PROGRESS_STATE_RUNNING;
            status->current_job_id = job.job_id;
            dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_JOB_START);

            DramTestParameters p = dram_make_params_from_work_item(
                job, reinterpret_cast<uint32_t>(&result_ring[slot]), expect_l1_addr, observe_l1_addr);

            const bool job_completed =
                run_one_dram_job(p, &result_ring[slot], expect_words, observe_words, status, ctrl, sync_mb);

            noc_async_write_barrier();

            if (!job_completed) {
                break;
            }

            ctrl->head = head + 1u;
            ctrl->jobs_completed++;
            status->jobs_completed = ctrl->jobs_completed;

            dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_JOB_DONE);
        }

        status->state = DRAM_PROGRESS_STATE_IDLE;
        dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_WAIT);

        for (int i = 0; i < 64; ++i) {
            noc_async_read_barrier();

            if (ctrl->head != ctrl->tail || ctrl->stop_requested) {
                break;
            }
        }

        if (ctrl->stop_requested) {
            break;
        }
    }

    sync_mb[MB_STOP] = 1u;
    sync_mb[MB_CURRENT_STAGE] = MB_STAGE_JOB_DONE;

    noc_async_write_barrier();

    status->state = DRAM_PROGRESS_STATE_DONE;
    dram_status_heartbeat(status, DRAM_PROGRESS_STAGE_DONE);
}
