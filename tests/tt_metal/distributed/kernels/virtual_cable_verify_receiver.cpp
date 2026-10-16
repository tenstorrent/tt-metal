// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Checks every frame one virtual-cable lane delivers into this core's ring, in order (test_virtual_cables --verify).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAYLOAD_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t FRAMES = get_compile_time_arg_val(1);
    constexpr uint32_t RING_ADDR = get_compile_time_arg_val(2);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(3);
    constexpr uint32_t STATUS_ADDR = get_compile_time_arg_val(4);  // {checked, corrupt, overrun, ready, abort}
    constexpr uint32_t WORDS = PAYLOAD_BYTES / sizeof(uint32_t);
    constexpr uint32_t kUnwritten = 0xFFFFFFFFu;  // the host's fill

    volatile tt_l1_ptr uint32_t* const status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(STATUS_ADDR);
    status[3] = 1;  // ready: the host releases the senders once every checker is
    for (uint32_t i = 0; i < FRAMES; ++i) {
        volatile tt_l1_ptr uint32_t* const f =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RING_ADDR + (i % SLOTS) * PAYLOAD_BYTES);
        // The slot holds an older frame (or the fill) until frame i's last word lands; the host aborts a stuck wait.
        uint32_t seen = f[WORDS - 1];
        while ((seen == kUnwritten || seen < i) && status[4] == 0) {
            invalidate_l1_cache();
            seen = f[WORDS - 1];
        }
        if (status[4] != 0) {
            return;
        }
        invalidate_l1_cache();  // the payload below, not just the word the spin read
        bool ok = seen == i && f[0] == i;
        for (uint32_t k = 3; k < WORDS - 1 && ok; ++k) {
            ok = f[k] == (0xC0DE0000u | (k & 0xFFFFu));
        }
        const uint32_t first = f[0];
        if (seen != i || f[WORDS - 1] != i || (first != kUnwritten && first > i)) {
            status[2] = status[2] + 1;  // a later frame overwrote the slot before it was checked
        } else if (!ok) {
            status[1] = status[1] + 1;
        }
        status[0] = i + 1;
    }
}
