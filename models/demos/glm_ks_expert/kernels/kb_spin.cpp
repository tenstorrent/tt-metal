// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 1 probe (NCRISC): drains the bench's outputs like kb_writer, but between checks spins as the flat expert's
// event loops do - invalidate_l1_cache() + volatile L1 reads of a few words - to see whether a polling RISC slows the
// matmul on the same core. RT: 0 NSB, 1 H, 2 P, 3 spin mode (0 none: plain cb_wait_front; 1 tight poll; 2 poll with
// invalidate_l1_cache), 4 an L1 word address to poll
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t nsb = get_arg_val<uint32_t>(0), nh = get_arg_val<uint32_t>(1), np = get_arg_val<uint32_t>(2);
    const uint32_t mode = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    uint32_t sink = 0;
    for (uint32_t sb = 0; sb < nsb; ++sb) {
        if (np) {
            while (mode && !cb_pages_available_at_front(17, np)) {
                if (mode == 2) {
                    invalidate_l1_cache();
                }
                sink += w[0] + w[1] + w[2] + w[3];
            }
            cb_wait_front(17, np);
            cb_pop_front(17, np);
        }
        while (mode && !cb_pages_available_at_front(16, nh)) {
            if (mode == 2) {
                invalidate_l1_cache();
            }
            sink += w[0] + w[1] + w[2] + w[3];
        }
        cb_wait_front(16, nh);
        cb_pop_front(16, nh);
    }
    w[8] = sink;
}
