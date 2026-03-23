#ifndef _DRAM_COMMON_H
#define _DRAM_COMMON_H

#include <stdint.h>

#include "patterns/patterns.hpp"

#define DRAM_TEST_NOC_WORD_BYTES 64
#define DRAM_TEST_MAX_BANK_BYTES 0xFF000000ULL

struct DramBaseResult {
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
    uint64_t prepare_ticks;
    uint64_t write_ticks;
    uint64_t read_ticks;
};

#define TEST_PARAMETERS(X)         \
    X(uint32_t, bank_id)           \
    X(uint32_t, bank_offset_lo)    \
    X(uint32_t, bank_offset_hi)    \
    X(uint32_t, total_bytes)       \
    X(uint32_t, chunk_bytes)       \
    X(uint32_t, pattern_id)        \
    X(uint32_t, seed)              \
    X(uint32_t, pass_index)        \
    X(uint32_t, repeat_index)      \
    X(uint32_t, result_l1_addr)    \
    X(uint32_t, expect_l1_addr)    \
    X(uint32_t, observe_l1_addr)   \
    X(uint32_t, write_noc)         \
    X(uint32_t, read_noc)          \
    X(uint32_t, max_burst_len)     \
    X(uint32_t, transfer_len_mode) \
    X(uint32_t, skip_writes)       \
    X(uint32_t, skip_reads)

struct DramTestParameters {
#define X(t, x) t x;
    TEST_PARAMETERS(X)
#undef X
};

static inline uint64_t dram_test_bank_offset(const DramTestParameters& p) {
    return ((uint64_t)p.bank_offset_hi << 32) | p.bank_offset_lo;
}

#endif /* _DRAM_COMMON_H */
