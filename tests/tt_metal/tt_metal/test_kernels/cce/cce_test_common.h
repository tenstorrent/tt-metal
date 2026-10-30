// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "experimental/cce_hart.h"
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#include "risc_common.h"

#include "cce_test_status_abi.h"

namespace cce_test {

using experimental::cce_hart::arg_u64;
using experimental::cce_hart::own_tl1_words;
using experimental::cce_hart::peer_tl1_word;
using experimental::cce_hart::read_cycle_counter;
using experimental::cce_hart::wait_until;
using experimental::cce_hart::WaitDeadline;

inline constexpr uint32_t kLineBytes = L2_CACHE_LINE_SIZE;
static_assert(kLineBytes == cce_test_status_abi::kLineBytes);

// The TL1 area tests may use: from the end of the harts' kernel text (the HAL's UNRESERVED base for the DRAM core
// before its alignment, qa_hal_dram.cpp:67-68) to the end of the hart-cacheable port.
inline constexpr uint64_t kTestAreaBegin =
    MEM_CCE_KERNEL_BASE + static_cast<uint64_t>(MEM_CCE_KERNEL_SIZE) * MEM_CCE_LOCAL_HARTS;
inline constexpr uint64_t kTestAreaEnd = MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE;

// A status line the host can find: 64 B aligned, inside the test area. Otherwise the kernel writes nothing and the host
// deadline reports it.
inline bool status_line_valid(uint32_t status_offset) {
    return status_offset % kLineBytes == 0 && status_offset >= kTestAreaBegin &&
           status_offset + static_cast<uint64_t>(kLineBytes) <= kTestAreaEnd;
}

// Ends a status line whose fields are stored: `fence w, w`, then the done magic.
inline void publish_done_magic(volatile uint32_t* status, uint32_t done_magic) {
    asm volatile("fence w, w" ::: "memory");
    status[cce_test_status_abi::kStatusMagic] = done_magic;
}

inline __attribute__((always_inline)) void copy_words(
    volatile uint32_t* dst, const volatile uint32_t* src, uint32_t num_words, uint32_t& xor_accumulator) {
    for (uint32_t i = 0; i < num_words; i++) {
        const uint32_t value = src[i];
        dst[i] = value;
        xor_accumulator ^= value;
    }
}

inline bool on_one_line(uint64_t first_address, uint64_t second_address) {
    return first_address / kLineBytes == second_address / kLineBytes;
}

// Spins until `ready()` or `budget_cycles` pass; returns whether it became ready and stores the cycles spent.
template <typename Ready>
inline bool poll_bounded(Ready ready, uint64_t budget_cycles, uint32_t& cycles_spent) {
    const uint64_t start = read_cycle_counter();
    const WaitDeadline deadline(budget_cycles);
    const bool is_ready = wait_until(deadline, ready);
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
