// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// CB pointer validation utility for verifying CB pointer wrapping in fused kernels.
// Used to check that all CB read/write pointers return to their initial addresses
// after one iteration, which is a prerequisite for adding looping.
//
// Layout per capture (6 * NUM_CBS uint32_t):
//   [0]           BRISC rd[0..63]
//   [NUM_CBS]     BRISC wr[0..63]
//   [2*NUM_CBS]   NCRISC rd[0..63]
//   [3*NUM_CBS]   NCRISC wr[0..63]
//   [4*NUM_CBS]   UNPACK rd[0..63]
//   [5*NUM_CBS]   PACK wr[0..63]
// AFTER section starts at offset SECTION_SIZE (6*NUM_CBS).

#pragma once

#include <cstdint>

namespace cb_validation {

constexpr uint32_t NUM_CBS = 64;
constexpr uint32_t SECTION_SIZE = 6 * NUM_CBS;  // uint32_t per before/after section
constexpr uint32_t TOTAL_UINT32 = 2 * SECTION_SIZE;

// Capture CB pointers for all RISCs into the given L1 address.
// phase_offset: 0 for BEFORE, SECTION_SIZE for AFTER.
FORCE_INLINE void capture(uint32_t addr, uint32_t phase_offset) {
    volatile uint32_t* base = reinterpret_cast<volatile uint32_t*>(addr);

#if defined(COMPILE_FOR_BRISC)
    for (uint32_t i = 0; i < NUM_CBS; i++) {
        base[phase_offset + i] = get_read_ptr(i);
        base[phase_offset + NUM_CBS + i] = get_write_ptr(i);
    }
#elif defined(COMPILE_FOR_NCRISC)
    for (uint32_t i = 0; i < NUM_CBS; i++) {
        base[phase_offset + 2 * NUM_CBS + i] = get_read_ptr(i);
        base[phase_offset + 3 * NUM_CBS + i] = get_write_ptr(i);
    }
#elif defined(COMPILE_FOR_TRISC)
    UNPACK(({
        for (uint32_t i = 0; i < NUM_CBS; i++) {
            base[phase_offset + 4 * NUM_CBS + i] = get_local_cb_interface(i).fifo_rd_ptr;
        }
    }));
    PACK(({
        for (uint32_t i = 0; i < NUM_CBS; i++) {
            base[phase_offset + 5 * NUM_CBS + i] = get_local_cb_interface(i).fifo_wr_ptr;
        }
    }));
#endif
}

}  // namespace cb_validation
