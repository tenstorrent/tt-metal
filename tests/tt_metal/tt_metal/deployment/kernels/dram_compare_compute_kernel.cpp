// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "patterns/sync_mailbox.hpp"
#include "patterns/dram_pattern_fill.hpp"
#include "timestamp.hpp"

static inline void prof_add_u64(volatile tt_l1_ptr uint32_t* mb, uint32_t lo_idx, uint64_t delta) {
    const uint64_t cur = (static_cast<uint64_t>(mb[lo_idx + 1u]) << 32) | static_cast<uint64_t>(mb[lo_idx]);
    const uint64_t nxt = cur + delta;
    mb[lo_idx] = static_cast<uint32_t>(nxt);
    mb[lo_idx + 1u] = static_cast<uint32_t>(nxt >> 32);
}

static inline uint32_t prof_generate_counter_idx(uint32_t helper_id) {
    return (helper_id == DRAM_COMPARE_HELPER_MATH) ? MB_PROF_MATH_GEN_ACTIVE_LO : MB_PROF_PACK_GEN_ACTIVE_LO;
}

static inline uint32_t prof_compare_counter_idx(uint32_t helper_id) {
    if (helper_id == DRAM_COMPARE_HELPER_MATH) {
        return MB_PROF_MATH_CMP_ACTIVE_LO;
    }
    if (helper_id == DRAM_COMPARE_HELPER_PACK) {
        return MB_PROF_PACK_CMP_ACTIVE_LO;
    }
    return MB_PROF_UNPACK_CMP_ACTIVE_LO;
}

static inline void compare_range_and_publish(
    volatile tt_l1_ptr uint32_t* mb, uint32_t helper_id, uint32_t done_tag, uint32_t start_word, uint32_t end_word) {
    const uint32_t expected_l1_addr = mb[MB_COMPARE_SOURCE_L1_ADDR];
    const uint32_t observed_l1_addr = mb[MB_COMPARE_OBSERVED_L1_ADDR];
    const uint32_t base_byte_offset = mb[MB_COMPARE_BASE_BYTE_OFFSET];

    volatile tt_l1_ptr uint32_t* expected_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(expected_l1_addr);

    volatile tt_l1_ptr uint32_t* observed_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(observed_l1_addr);

    uint32_t result_idx = MB_COMPARE_MATH_RESULT;
    uint32_t done_idx = MB_COMPARE_MATH_DONE;
    uint32_t first_addr_idx = MB_COMPARE_MATH_FIRST_ADDR;
    uint32_t first_expected_idx = MB_COMPARE_MATH_FIRST_EXPECTED;
    uint32_t first_observed_idx = MB_COMPARE_MATH_FIRST_OBSERVED;

    if (helper_id == DRAM_COMPARE_HELPER_PACK) {
        result_idx = MB_COMPARE_PACK_RESULT;
        done_idx = MB_COMPARE_PACK_DONE;
        first_addr_idx = MB_COMPARE_PACK_FIRST_ADDR;
        first_expected_idx = MB_COMPARE_PACK_FIRST_EXPECTED;
        first_observed_idx = MB_COMPARE_PACK_FIRST_OBSERVED;
    } else if (helper_id == DRAM_COMPARE_HELPER_UNPACK) {
        result_idx = MB_COMPARE_UNPACK_RESULT;
        done_idx = MB_COMPARE_UNPACK_DONE;
        first_addr_idx = MB_COMPARE_UNPACK_FIRST_ADDR;
        first_expected_idx = MB_COMPARE_UNPACK_FIRST_EXPECTED;
        first_observed_idx = MB_COMPARE_UNPACK_FIRST_OBSERVED;
    }

    const uint64_t prof_t0 = timestamp();

    uint32_t failures = 0u;
    uint32_t first_addr = 0xFFFFFFFFu;
    uint32_t first_expected = 0u;
    uint32_t first_observed = 0u;

    for (uint32_t i = start_word; i < end_word; ++i) {
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

    mb[result_idx] = failures;
    mb[first_addr_idx] = first_addr;
    mb[first_expected_idx] = first_expected;
    mb[first_observed_idx] = first_observed;
    prof_add_u64(mb, prof_compare_counter_idx(helper_id), timestamp() - prof_t0);
    mb[done_idx] = done_tag;
}

static inline void generate_range_and_publish(
    volatile tt_l1_ptr uint32_t* mb, uint32_t helper_id, uint32_t done_tag, uint32_t start_word, uint32_t end_word) {
    const uint32_t dst_l1_addr = mb[MB_GENERATE_L1_ADDR];

    tt_l1_ptr uint32_t* dst_words = reinterpret_cast<tt_l1_ptr uint32_t*>(dst_l1_addr);

    const uint64_t prof_t0 = timestamp();

    DramPatternFillParams fill{};
    fill.pattern_id = mb[MB_PATTERN_ID_GLOBAL];
    fill.seed = mb[MB_SEED_GLOBAL];
    fill.pass_index = mb[MB_PASS_INDEX_GLOBAL];
    fill.repeat_index = mb[MB_REPEAT_INDEX_GLOBAL];
    fill.base_word_index = mb[MB_GENERATE_BASE_WORD_INDEX];
    fill.word_count = mb[MB_GENERATE_WORD_COUNT];

    dram_fill_pattern_buffer_range(dst_words, fill, start_word, end_word);

    prof_add_u64(mb, prof_generate_counter_idx(helper_id), timestamp() - prof_t0);

    if (helper_id == DRAM_COMPARE_HELPER_MATH) {
        mb[MB_GENERATE_MATH_DONE] = done_tag;
    } else if (helper_id == DRAM_COMPARE_HELPER_PACK) {
        mb[MB_GENERATE_PACK_DONE] = done_tag;
    }
}

static inline void compare_helper_loop(uint32_t helper_id) {
    const uint32_t mailbox_l1_addr = get_compile_time_arg_val(0);

    volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_l1_addr);

    uint32_t seen_generate_tag = 0u;
    uint32_t seen_compare_tag = 0u;

    while (true) {
        invalidate_l1_cache();

        if (mb[MB_STOP] != 0u) {
            break;
        }

        const uint32_t generate_tag = mb[MB_GENERATE_START];

        if ((generate_tag != 0u) && (helper_id != DRAM_COMPARE_HELPER_UNPACK) && (generate_tag != seen_generate_tag)) {
            seen_generate_tag = generate_tag;

            const uint32_t word_count = mb[MB_GENERATE_WORD_COUNT];
            const uint32_t mid = word_count >> 1;

            if (helper_id == DRAM_COMPARE_HELPER_MATH) {
                generate_range_and_publish(mb, helper_id, generate_tag, 0u, mid);
            } else if (helper_id == DRAM_COMPARE_HELPER_PACK) {
                generate_range_and_publish(mb, helper_id, generate_tag, mid, word_count);
            }

            continue;
        }

        const uint32_t compare_tag = mb[MB_COMPARE_START];

        if ((compare_tag == 0u) || (compare_tag == seen_compare_tag)) {
            continue;
        }

        seen_compare_tag = compare_tag;

        const uint32_t word_count = mb[MB_COMPARE_WORD_COUNT];

        // NCRISC compares the first quarter after its read phase. The compute helpers
        // compare the remaining three quarters while BRISC only orchestrates.
        const uint32_t q1 = word_count >> 2;
        const uint32_t q2 = word_count >> 1;
        const uint32_t q3 = (word_count * 3u) >> 2;

        uint32_t start_word = q1;
        uint32_t end_word = q2;

        if (helper_id == DRAM_COMPARE_HELPER_PACK) {
            start_word = q2;
            end_word = q3;
        } else if (helper_id == DRAM_COMPARE_HELPER_UNPACK) {
            start_word = q3;
            end_word = word_count;
        }

        compare_range_and_publish(mb, helper_id, compare_tag, start_word, end_word);
    }
}

void kernel_main() {
    MATH({ compare_helper_loop(DRAM_COMPARE_HELPER_MATH); });

    PACK({ compare_helper_loop(DRAM_COMPARE_HELPER_PACK); });

    UNPACK({ compare_helper_loop(DRAM_COMPARE_HELPER_UNPACK); });
}
