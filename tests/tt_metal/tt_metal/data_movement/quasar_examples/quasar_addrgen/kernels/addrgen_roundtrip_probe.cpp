// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal diagnostic for addrgen_1's source side. Reports three peeks (non-advancing reads):
//   [0] after reset only
//   [1] after setting only the inner-loop start offset (kInnerOffset)
//   [2] after additionally setting base_start (kBase), which lives in the *command buffer's* SRC_BASE
//       register rather than an addrgen register -- tells us whether the value the addrgen hands back
//       (peek/pop) already includes SRC_BASE, or whether SRC_BASE is only applied when pushing into the
//       command buffer.
// SRC_BASE is restored to 0 afterwards so later software reads on command buffer 1 are unaffected.
//
// Runtime args:
//   report_addr - L1 address to write the three 64-bit peeks to (6 words, low then high)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    constexpr uint64_t kInnerOffset = 0x1000;
    constexpr uint64_t kBase = 0x1234'5678'0000ULL;

    uint64_t peeks[3];
    overlay::reset_addrgen<overlay::ADDRGEN_1>();
    peeks[0] = overlay::peek_src_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_src_inner_loop_addrgen<overlay::ADDRGEN_1>(
        /*stride=*/0x100, /*end=*/0x100000, /*start=*/kInnerOffset);
    peeks[1] = overlay::peek_src_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_src_base_start_addrgen<overlay::ADDRGEN_1>(kBase);
    peeks[2] = overlay::peek_src_addrgen<overlay::ADDRGEN_1>();
    overlay::setup_src_base_start_addrgen<overlay::ADDRGEN_1>(0);

    DEVICE_PRINT("addrgen probe: reset 0x{:x} inner 0x{:x} inner+base 0x{:x}\n", peeks[0], peeks[1], peeks[2]);

    // Quasar DM stores go through the data cache; report through the uncached alias so the host sees it.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    for (uint32_t i = 0; i < 3; ++i) {
        report[2 * i] = static_cast<uint32_t>(peeks[i]);
        report[2 * i + 1] = static_cast<uint32_t>(peeks[i] >> 32);
    }
}
