// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "experimental/cce_remote_cb.h"
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"

namespace cce_test {

using experimental::cce_remote_cb::read_cycle_counter;
using experimental::cce_remote_cb::WaitDeadline;

inline constexpr uint32_t kLineBytes = L2_CACHE_LINE_SIZE;

inline __attribute__((always_inline)) void copy_words(
    volatile uint32_t* dst, const volatile uint32_t* src, uint32_t num_words, uint32_t& xor_accumulator) {
    for (uint32_t i = 0; i < num_words; i++) {
        const uint32_t value = src[i];
        dst[i] = value;
        xor_accumulator ^= value;
    }
}

inline uint64_t arg_u64(uint32_t lo_index, uint32_t hi_index) {
    return (static_cast<uint64_t>(get_arg_val<uint32_t>(hi_index)) << 32) | get_arg_val<uint32_t>(lo_index);
}

// Words of this CCE's TL1 that another agent (the host or another hart) writes or reads.
template <typename T = uint32_t>
inline volatile T* own_tl1_words(uint64_t tl1_offset) {
    return reinterpret_cast<volatile T*>(tl1_offset + static_cast<uint64_t>(MEM_L1_UNCACHED_BASE));
}

inline bool on_one_line(uint64_t first_address, uint64_t second_address) {
    return first_address / kLineBytes == second_address / kLineBytes;
}

// Spins until `ready()` or `budget_cycles` pass; returns whether it became ready and stores the cycles spent.
template <typename Ready>
inline bool poll_bounded(Ready ready, uint64_t budget_cycles, uint32_t& cycles_spent) {
    const uint64_t start = read_cycle_counter();
    const WaitDeadline deadline(budget_cycles);
    bool is_ready = ready();
    while (!is_ready && !deadline.expired()) {
        is_ready = ready();
    }
    cycles_spent = static_cast<uint32_t>(read_cycle_counter() - start);
    return is_ready;
}

// The mismatches of one data check: how many words differ, and the first of them.
struct DataCheck {
    uint32_t mismatched_words = 0;
    uint32_t first_mismatch_word = 0;
    uint32_t observed = 0;
    uint32_t expected = 0;

    void record(uint32_t word, uint32_t observed_value, uint32_t expected_value) {
        if (observed_value == expected_value) {
            return;
        }
        if (mismatched_words == 0) {
            first_mismatch_word = word;
            observed = observed_value;
            expected = expected_value;
        }
        mismatched_words++;
    }
};

}  // namespace cce_test
