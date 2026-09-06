// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads one DRAM core's RISCV_DEBUG_REG_WALL_CLOCK_L over the NoC as fast as it can, as a second agent on that
// core's debug timestamper latch.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint64_t kWallClockL = 0xFFB121F0ull;
enum Stat : uint32_t { kReadsLo = 0, kReadsHi = 1, kLast = 2, kFirst = 3, kStop = 4, kDone = 5 };
}  // namespace

void kernel_main() {
    const uint32_t drisc_x = get_arg_val<uint32_t>(0);
    const uint32_t drisc_y = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* stat = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    volatile tt_l1_ptr uint32_t* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);

    const uint64_t src = get_noc_addr(drisc_x, drisc_y, kWallClockL);
    uint64_t reads = 0;
    bool first = true;
    while (true) {
        noc_async_read(src, scratch, sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t v = *value;
        if (first) {
            stat[kFirst] = v;
            first = false;
        }
        reads++;
        if ((reads & 0x3FFu) == 0) {
            stat[kReadsLo] = static_cast<uint32_t>(reads);
            stat[kReadsHi] = static_cast<uint32_t>(reads >> 32);
            stat[kLast] = v;
            if (stat[kStop] != 0) {
                break;
            }
        }
    }
    stat[kReadsLo] = static_cast<uint32_t>(reads);
    stat[kReadsHi] = static_cast<uint32_t>(reads >> 32);
    stat[kDone] = 1;
}
