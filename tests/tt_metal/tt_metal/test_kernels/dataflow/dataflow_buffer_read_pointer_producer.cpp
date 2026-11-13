// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fills and publishes both entries of the bound `target` DFB. Entry 1 holds the pattern on the sender and zeros on
// every other node, so a receiver can tell whether the sender's multicast reached it.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer target(dfb::target);
    Noc noc;
    const bool is_sender = noc.is_local_bank(get_arg(args::sender_x), get_arg(args::sender_y));
    constexpr uint32_t num_words = get_arg(args::size_bytes) / sizeof(uint32_t);
    target.reserve_back(2);
    auto* entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(target.get_write_ptr());
    for (uint32_t i = 0; i < num_words; ++i) {
        entries[i] = 0;
        entries[num_words + i] = is_sender ? get_arg(args::pattern) + i : 0;
    }
    target.push_back(2);
}
