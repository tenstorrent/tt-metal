#include "args.hpp"
#include "common_dram.hpp"
#include "dram_utils.hpp"
#include "timestamp.hpp"
#include "patterns/patterns.hpp"

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    DramTestParameters p;
    /* Fills `p` with runtime arguments */
    ARG_INIT_PARAMS(TEST_PARAMETERS);

    const uint64_t bank_offset_base = dram_test_bank_offset(p);

    volatile DramBaseResult* result = (volatile DramBaseResult*)p.result_l1_addr;

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

    // Basic sanity checks for the first port
    if ((p.chunk_bytes == 0u) || ((p.chunk_bytes & 0x3u) != 0u) || ((p.total_bytes & 0x3u) != 0u)) {
        result->failures = 1u;
        result->first_fail_addr = 0u;
        result->first_expected = 0u;
        result->first_observed = p.chunk_bytes;
        return;
    }

    uint32_t* expect_words = (uint32_t*)p.expect_l1_addr;
    uint32_t* observe_words = (uint32_t*)p.observe_l1_addr;
    uint64_t total_prepare_ticks = 0u;
    uint64_t total_write_ticks = 0u;
    uint64_t total_read_ticks = 0u;

    uint32_t rng_state = p.seed ^ p.pass_index;
    if (rng_state == 0u) {
        rng_state = 1u;
    }

    DramXoshiro128ppState xoshiro_state = dram_pattern_random_xoshiro128pp_init(p.seed ^ p.pass_index);

    const bool reuse_checkerboard_buffer = (p.pattern_id == DRAM_PATTERN_CHECKERBOARD);

    if (reuse_checkerboard_buffer) {
        const uint32_t precompute_word_count = p.chunk_bytes / sizeof(uint32_t);
        uint64_t t0 = timestamp();
        dram_pattern_checkerboard_fill_buffer(expect_words, precompute_word_count, p.pass_index);
        uint64_t t1 = timestamp();
        total_prepare_ticks += (t1 - t0);
    }

    for (uint32_t offset = 0; offset < p.total_bytes;) {
        uint32_t remaining_bytes = p.total_bytes - offset;
        uint32_t transfer_bytes = dram_choose_transfer_len(p, remaining_bytes, rng_state);
        const uint32_t word_count = transfer_bytes / sizeof(uint32_t);
        const uint32_t base_word_index = offset / sizeof(uint32_t);

        if (!reuse_checkerboard_buffer) {
            uint64_t t0 = timestamp();

            for (uint32_t i = 0; i < word_count; ++i) {
                if (p.pattern_id == DRAM_PATTERN_RANDOM) {
                    rng_state = dram_pattern_random_step(rng_state);
                    expect_words[i] = rng_state;
                } else if (p.pattern_id == DRAM_PATTERN_RANDOM_XOSHIRO128PP) {
                    expect_words[i] = dram_pattern_random_xoshiro128pp_next(xoshiro_state);
                } else {
                    expect_words[i] =
                        dram_pattern_generate(p.pattern_id, p.seed, p.pass_index, base_word_index + i, p.repeat_index);
                }
            }

            uint64_t t1 = timestamp();
            total_prepare_ticks += (t1 - t0);
        }

        if (!p.skip_writes) {
            uint64_t write_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                p.bank_id, (uint32_t)(bank_offset_base + (uint64_t)offset), p.write_noc);

            uint64_t t0 = timestamp();
            noc_async_write(p.expect_l1_addr, write_dram_noc_addr, transfer_bytes);
            noc_async_write_barrier();
            uint64_t t1 = timestamp();
            total_write_ticks += (t1 - t0);
        }

        if (!p.skip_reads) {
            uint64_t read_dram_noc_addr =
                get_noc_addr_from_bank_id<true>(p.bank_id, (uint32_t)(bank_offset_base + (uint64_t)offset), p.read_noc);

            uint64_t t0 = timestamp();
            noc_async_read(read_dram_noc_addr, p.observe_l1_addr, transfer_bytes);
            noc_async_read_barrier();
            uint64_t t1 = timestamp();
            total_read_ticks += (t1 - t0);

            for (uint32_t i = 0; i < word_count; ++i) {
                uint32_t expected = expect_words[i];
                uint32_t observed = observe_words[i];

                result->words_checked++;

                if (observed != expected) {
                    if (result->failures == 0u) {
                        result->first_fail_addr = offset + i * sizeof(uint32_t);
                        result->first_expected = expected;
                        result->first_observed = observed;
                    }
                    result->failures++;
                }
            }
        }

        offset += transfer_bytes;
        result->transfers++;
    }

    result->prepare_ticks = total_prepare_ticks;
    result->write_ticks = total_write_ticks;
    result->read_ticks = total_read_ticks;
}
