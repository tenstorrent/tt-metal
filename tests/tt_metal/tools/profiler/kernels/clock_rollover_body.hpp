// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "clock_rollover_layout.hpp"
#ifndef PROFILE_KERNEL
#error "clock_rollover requires TT_METAL_DEVICE_PROFILER=1"
#endif
#ifndef ARCH_BLACKHOLE
#error "clock_rollover currently supports Blackhole only"
#endif
namespace clock_rollover {
struct Clock {
    uint32_t hi;
    uint32_t lo;
};
inline __attribute__((always_inline)) Clock reference(uint32_t& retries) {
    volatile tt_reg_ptr uint32_t* reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while (true) {
        uint32_t hi = reg[1];
        uint32_t lo = reg[0];
        uint32_t hi_after = reg[1];
        if (hi == hi_after) {
            return {hi, lo};
        }
        ++retries;
    }
}
inline uint64_t wide(Clock c) { return (uint64_t(c.hi) << 32) | c.lo; }
inline void run() {
    using namespace kernel_profiler;
    const uint32_t lane = myRiscID;
    ASSERT(lane < lanes);
    if (lane >= lanes) {
        return;
    }
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CLOCK_ROLLOVER_ADDRESS) + lane * lane_words;
    for (uint32_t i = 0; i < lane_words; ++i) {
        out[i] = 0;
    }
    out[0] = magic;
    out[1] = version;
    out[2] = lane;
    out[3] = capacity;
    constexpr uint32_t scratch = PROFILER_L1_VECTOR_SIZE - 4;
    ASSERT(wIndex + stackSize < scratch);
    if (wIndex + stackSize >= scratch) {
        out[10] = 2;
        return;
    }
    uint32_t retries = 0;
    const Clock entered = reference(retries);
    Clock now = entered;
    while (now.lo < 0xfffff000u) {
        now = reference(retries);
        ASSERT(wide(now) - wide(entered) < (uint64_t(2) << 32));
        if (wide(now) - wide(entered) >= (uint64_t(2) << 32)) {
            out[10] = 3;
            return;
        }
    }
    const uint32_t initial_hi = now.hi;
    out[6] = initial_hi;
    out[11] = now.hi;
    out[12] = now.lo;
    uint32_t total = 0;
    uint32_t transition = 0xffffffffu;
    while (true) {
        ASSERT(total < 1000000u);
        if (total >= 1000000u) {
            out[10] = 4;
            return;
        }
        const uint32_t slot = total % capacity;
        volatile tt_l1_ptr uint32_t* record = out + header_words + slot * record_words;
        const uint32_t retries_before = retries;
        const Clock before = reference(retries);
        // These are the actual production writer, including its packed L1 stores.
        mark_time_at_index_inlined(scratch, 0);
        mark_time_at_index_inlined(scratch + 2, 0);
        const Clock after = reference(retries);
        record[0] = total;
        record[1] = slot;
        record[2] = before.hi;
        record[3] = before.lo;
        record[4] = profiler_data_buffer[lane].data[scratch];
        record[5] = profiler_data_buffer[lane].data[scratch + 1];
        record[6] = profiler_data_buffer[lane].data[scratch + 2];
        record[7] = profiler_data_buffer[lane].data[scratch + 3];
        record[8] = after.hi;
        record[9] = after.lo;
        record[10] = retries - retries_before;
        record[11] = 0;
        if (transition == 0xffffffffu && after.hi != initial_hi) {
            transition = total;
        }
        ++total;
        now = after;
        if (transition != 0xffffffffu && total > transition + 64) {
            break;
        }
    }
    out[4] = total;
    out[5] = total % capacity;
    out[7] = now.hi;
    out[8] = transition;
    out[9] = retries;
    out[13] = now.hi;
    out[14] = now.lo;
    out[10] = 1;
}
}  // namespace clock_rollover
