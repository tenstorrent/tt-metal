// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ping-pong between CCE0 and CCE1 of one tile with peer_signal / peer_wait, flags in CCE0's SRAM. The sender (role 0)
// is hart 0 of CCE0, the responder (role 1) hart 0 of CCE1.
//
// Runtime args: 0 = role, 1 = rounds, 2 = flag A offset (sender -> responder), 3 = flag B offset (responder ->
// sender), 4 = status block offset (own SRAM). Status: 1 rounds completed, 4 done.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/keraunos_cce.h"

void kernel_main() {
    using namespace experimental::keraunos;
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t rounds = get_arg_val<uint32_t>(1);
    const uint32_t flag_a = get_arg_val<uint32_t>(2);
    const uint32_t flag_b = get_arg_val<uint32_t>(3);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(get_arg_val<uint32_t>(4)));
    status[0] = 0x1D3A0001;
    for (uint32_t round = 1; round <= rounds; round++) {
        if (role == 0) {
            peer_signal(flag_a, round);
            peer_wait(flag_b, round);
        } else {
            peer_wait(flag_a, round);
            peer_signal(flag_b, round);
        }
        status[1] = round;
    }
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
