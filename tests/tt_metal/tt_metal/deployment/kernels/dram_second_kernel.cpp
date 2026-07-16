// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common_dram.hpp"
#include "dram_utils.hpp"
#include "patterns/sync_mailbox.hpp"
#include "timestamp.hpp"

// NCRISC_OBSERVED_CORRUPTION_PATCH
// Error insertion belongs to the NCRISC read path. NCRISC reads DDR into observed L1,
// then deliberately corrupts observed words before compare consumes them.

// 0 = disabled, 1 = enabled
#define INSERT_WRITE_ERRORS 1
#define INSERT_READ_ERRORS 1

// Number of observed words to corrupt after each NCRISC read.
#define WRITE_ERROR_COUNT 20u
#define READ_ERROR_COUNT 20u

// Optional selection knobs live here, not in BRISC.
#define WRITE_ERROR_STRIDE_WORDS 262144u
#define READ_ERROR_STRIDE_WORDS 393216u
#define WRITE_ERROR_START_WORD 0u
#define READ_ERROR_START_WORD 20u

static inline bool ncrisc_should_insert_write_error(uint32_t global_word_index) {
#if INSERT_WRITE_ERRORS
    (void)global_word_index;
    return true;
#else
    (void)global_word_index;
    return false;
#endif
}

static inline bool ncrisc_should_insert_read_error(uint32_t global_word_index) {
#if INSERT_READ_ERRORS
    (void)global_word_index;
    return true;
#else
    (void)global_word_index;
    return false;
#endif
}

static inline void corrupt_observed_words_after_ncrisc_read(
    uint32_t observe_l1_addr,
    volatile tt_l1_ptr uint32_t* mb,
    uint32_t current_pattern_id,
    uint32_t base_word_index,
    uint32_t word_count) {
    const uint32_t selected_pattern_id = mb[MB_INSERT_ERRORS_PATTERN_ID];

    if (selected_pattern_id == 0u) {
        return;
    }

    if (mb[MB_INSERT_ERRORS_DONE] != 0u) {
        return;
    }

    if (selected_pattern_id != current_pattern_id) {
        return;
    }

    // Inject only once, at the first pass/repeat and only on DRAM bank 0.
    if (mb[MB_REPEAT_INDEX_GLOBAL] != 0u) {
        return;
    }

    if (mb[MB_PASS_INDEX_GLOBAL] != 0u) {
        return;
    }

    if (mb[MB_BANK_ID] != 0u) {
        return;
    }

    // Only corrupt the beginning of the selected pattern/job.
    if (base_word_index != 0u) {
        return;
    }

    // Mark done before writing observed words so this mailbox cannot inject again.
    mb[MB_INSERT_ERRORS_DONE] = 1u;

    // Local L1 overwrite only. No NoC operation is issued here.
    volatile tt_l1_ptr uint32_t* observed_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(observe_l1_addr);

    for (uint32_t i = 0; (i < WRITE_ERROR_COUNT) && (i < word_count); ++i) {
        observed_words[i] = 0xDEADBEEFu ^ i;
    }
}

static inline void prof_add_u64(volatile tt_l1_ptr uint32_t* mb, uint32_t lo_idx, uint64_t delta) {
    const uint64_t cur = (static_cast<uint64_t>(mb[lo_idx + 1u]) << 32) | static_cast<uint64_t>(mb[lo_idx]);
    const uint64_t nxt = cur + delta;
    mb[lo_idx] = static_cast<uint32_t>(nxt);
    mb[lo_idx + 1u] = static_cast<uint32_t>(nxt >> 32);
}

static inline uint32_t dram_ncrisc_read_word_from_bank(
    uint32_t bank_id, uint64_t bank_offset_base, uint32_t byte_offset, uint32_t noc_id, uint32_t scratch_l1_addr) {
    const uint32_t aligned_offset = byte_offset & ~0x1Fu;
    const uint32_t word_index = (byte_offset & 0x1Fu) / sizeof(uint32_t);
    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(
        bank_id, static_cast<uint32_t>(bank_offset_base + static_cast<uint64_t>(aligned_offset)), noc_id);

    noc_async_read(dram_noc_addr, scratch_l1_addr, 32);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* scratch_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1_addr);
    return scratch_words[word_index];
}

static inline void dram_ncrisc_compare_first_quarter(
    volatile tt_l1_ptr uint32_t* mb,
    uint32_t start_tag,
    uint32_t expected_l1_addr,
    uint32_t observed_l1_addr,
    uint32_t base_byte_offset,
    uint32_t word_count) {
    if (word_count == 0u) {
        mb[MB_COMPARE_NCRISC_RESULT] = 0u;
        mb[MB_COMPARE_NCRISC_FIRST_ADDR] = 0xFFFFFFFFu;
        mb[MB_COMPARE_NCRISC_FIRST_EXPECTED] = 0u;
        mb[MB_COMPARE_NCRISC_FIRST_OBSERVED] = 0u;
        mb[MB_COMPARE_NCRISC_DONE] = start_tag;
        return;
    }

    // Important: use the command snapshot captured when NCRISC accepted the
    // start tag. BRISC may update mailbox active/current fields to prepare or
    // launch the next chunk while NCRISC is still doing read + compare for this
    // command. Reading MB_CURRENT_* or MB_GEN_ACTIVE_* here would make the
    // NCRISC compare worker occasionally compare the observed data for one
    // chunk against the expected buffer/offset for another chunk.
    const uint32_t end_word = word_count >> 2;

    invalidate_l1_cache();

    volatile tt_l1_ptr uint32_t* expected_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(expected_l1_addr);
    volatile tt_l1_ptr uint32_t* observed_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(observed_l1_addr);

    uint32_t failures = 0u;
    uint32_t first_addr = 0xFFFFFFFFu;
    uint32_t first_expected = 0u;
    uint32_t first_observed = 0u;

    for (uint32_t i = 0u; i < end_word; ++i) {
        const uint32_t expected = expected_words[i];
        const uint32_t observed = observed_words[i];
        if (observed != expected) {
            if (failures == 0u) {
                first_addr = base_byte_offset + i * sizeof(uint32_t);
                first_expected = expected;
                first_observed = observed;
            }
            failures++;
        }
    }

    mb[MB_COMPARE_NCRISC_RESULT] = failures;
    mb[MB_COMPARE_NCRISC_FIRST_ADDR] = first_addr;
    mb[MB_COMPARE_NCRISC_FIRST_EXPECTED] = first_expected;
    mb[MB_COMPARE_NCRISC_FIRST_OBSERVED] = first_observed;
    mb[MB_COMPARE_NCRISC_DONE] = start_tag;
}

static inline void dram_ncrisc_run_deferred_diag(
    volatile tt_l1_ptr uint32_t* mb,
    uint32_t bank_id,
    uint64_t bank_offset_base,
    uint32_t read_noc,
    uint32_t scratch_l1_addr) {
    if (mb[MB_NCRISC_DIAG_REQUEST] == 0u) {
        return;
    }

    const uint64_t diag_t0 = timestamp();

    const uint32_t fail_byte_offset = mb[MB_NCRISC_DIAG_ADDR];
    const uint32_t expected = mb[MB_NCRISC_DIAG_EXPECTED];

    const uint32_t reread0 =
        dram_ncrisc_read_word_from_bank(bank_id, bank_offset_base, fail_byte_offset, read_noc, scratch_l1_addr);
    const uint32_t reread1 =
        dram_ncrisc_read_word_from_bank(bank_id, bank_offset_base, fail_byte_offset, read_noc, scratch_l1_addr);
    const uint32_t reread2 =
        dram_ncrisc_read_word_from_bank(bank_id, bank_offset_base, fail_byte_offset, read_noc, scratch_l1_addr);
    const uint32_t reread3 =
        dram_ncrisc_read_word_from_bank(bank_id, bank_offset_base, fail_byte_offset, read_noc, scratch_l1_addr);
    const uint32_t reread4 =
        dram_ncrisc_read_word_from_bank(bank_id, bank_offset_base, fail_byte_offset, read_noc, scratch_l1_addr);

    const bool all_same = (reread0 == reread1) && (reread0 == reread2) && (reread0 == reread3) && (reread0 == reread4);

    uint32_t classified_kind = DRAM_FAILURE_READ;
    if (all_same && reread0 != expected) {
        classified_kind = DRAM_FAILURE_WRITE;
    }

    mb[MB_NCRISC_DIAG_READBACK0] = reread0;
    mb[MB_NCRISC_DIAG_READBACK1] = reread1;
    mb[MB_NCRISC_DIAG_READBACK2] = reread2;
    mb[MB_NCRISC_DIAG_READBACK3] = reread3;
    mb[MB_NCRISC_DIAG_READBACK4] = reread4;
    mb[MB_NCRISC_DIAG_KIND] = classified_kind;
    prof_add_u64(mb, MB_PROF_NCRISC_DIAG_ACTIVE_LO, timestamp() - diag_t0);
    mb[MB_NCRISC_DIAG_REQUEST] = 0u;
}

void kernel_main() {
    const uint32_t sync_mailbox_l1_addr = get_arg_val<uint32_t>(0);

    volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_mailbox_l1_addr);

    if (mb[MB_MAGIC] != DRAM_SYNC_MAILBOX_MAGIC) {
        return;
    }

    uint32_t seen_start_tag = 0u;
    uint64_t idle_t0 = timestamp();

    while (true) {
        noc_async_read_barrier();

        if (mb[MB_STOP] != 0u) {
            break;
        }

        const uint32_t start_tag = mb[MB_NCRISC_START];

        if ((start_tag == 0u) || (start_tag == seen_start_tag)) {
            continue;
        }

        prof_add_u64(mb, MB_PROF_NCRISC_IDLE_LO, timestamp() - idle_t0);
        seen_start_tag = start_tag;

        const uint32_t bank_id = mb[MB_BANK_ID];
        const uint64_t bank_offset_base =
            (static_cast<uint64_t>(mb[MB_BANK_OFFSET_HI]) << 32) | static_cast<uint64_t>(mb[MB_BANK_OFFSET_LO]);

        const uint32_t offset = mb[MB_CURRENT_OFFSET_BYTES];
        const uint32_t transfer_bytes = mb[MB_CURRENT_TRANSFER_BYTES];

        const uint32_t expect_l1_addr = mb[MB_GEN_ACTIVE_L1_ADDR];
        const uint32_t observe_l1_addr = mb[MB_OBS_ACTIVE_L1_ADDR];

        const uint32_t write_noc = mb[MB_WRITE_NOC];
        const uint32_t read_noc = mb[MB_READ_NOC];

        const uint32_t skip_writes = mb[MB_SKIP_WRITES];
        const uint32_t skip_reads = mb[MB_SKIP_READS];

        mb[MB_NCRISC_ERROR] = MB_ERROR_NONE;
        mb[MB_COMPARE_NCRISC_DONE] = 0u;
        mb[MB_COMPARE_NCRISC_RESULT] = 0u;
        mb[MB_COMPARE_NCRISC_FIRST_ADDR] = 0xFFFFFFFFu;
        mb[MB_COMPARE_NCRISC_FIRST_EXPECTED] = 0u;
        mb[MB_COMPARE_NCRISC_FIRST_OBSERVED] = 0u;
        mb[MB_NCRISC_ACTIVE_OFFSET_BYTES] = offset;
        mb[MB_NCRISC_ACTIVE_TRANSFER_BYTES] = transfer_bytes;

        const uint32_t diag_request = mb[MB_NCRISC_DIAG_REQUEST];
        const bool diag_only = (transfer_bytes == 0u) && (diag_request != 0u);

        if ((!diag_only && transfer_bytes == 0u) || ((transfer_bytes & 0x3u) != 0u)) {
            mb[MB_NCRISC_ERROR] = MB_ERROR_NCRISC_BAD_TRANSFER;
            mb[MB_ERROR] = MB_ERROR_NCRISC_BAD_TRANSFER;
            mb[MB_NCRISC_DONE] = start_tag;
            idle_t0 = timestamp();
            continue;
        }

        if (!diag_only && !skip_writes) {
            const uint64_t write_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                bank_id, static_cast<uint32_t>(bank_offset_base + static_cast<uint64_t>(offset)), write_noc);

            const uint64_t write_t0 = timestamp();
            noc_async_write(expect_l1_addr, write_dram_noc_addr, transfer_bytes);
            // Protect real DDR write over NoC. Do not start DDR readback until this is complete.
            noc_async_write_barrier();
            prof_add_u64(mb, MB_PROF_NCRISC_WRITE_ACTIVE_LO, timestamp() - write_t0);
        }

        if (!diag_only && !skip_reads) {
            const uint64_t read_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                bank_id, static_cast<uint32_t>(bank_offset_base + static_cast<uint64_t>(offset)), read_noc);

            const uint64_t read_t0 = timestamp();
            noc_async_read(read_dram_noc_addr, observe_l1_addr, transfer_bytes);
            // Protect real DDR read over NoC. observed L1 is valid only after this barrier.
            noc_async_read_barrier();

            // After DDR -> observed L1 is complete, overwrite first words locally in L1.
            corrupt_observed_words_after_ncrisc_read(
                observe_l1_addr,
                mb,
                mb[MB_PATTERN_ID_GLOBAL],
                offset / sizeof(uint32_t),
                transfer_bytes / sizeof(uint32_t));

            prof_add_u64(mb, MB_PROF_NCRISC_READ_ACTIVE_LO, timestamp() - read_t0);

            // Compare the first quarter while data is hot in L1. BRISC consumes this
            // result immediately after waiting for MB_NCRISC_DONE.
            dram_ncrisc_compare_first_quarter(
                mb, start_tag, expect_l1_addr, observe_l1_addr, offset, transfer_bytes / sizeof(uint32_t));
        }

        // Deferred diagnostic from the previous compared chunk. This is intentionally
        // tied to the next NCRISC command so BRISC does not perform inline re-reads.
        dram_ncrisc_run_deferred_diag(mb, bank_id, bank_offset_base, read_noc, observe_l1_addr);

        mb[MB_NCRISC_DONE] = start_tag;
        idle_t0 = timestamp();
    }
}
