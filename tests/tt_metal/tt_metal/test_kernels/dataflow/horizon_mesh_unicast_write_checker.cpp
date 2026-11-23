// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified Mesh-API checker kernel.
// Unicast (noc_packet_type != 4): spins on recv_buf_a[0] != 0, verifies chunk_words_a
// words against base_value+i, writes pass/fail.
// Scatter (noc_packet_type == 4): additionally verifies recv_buf_b (second chunk).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

static constexpr uint32_t HORIZON_NOC_UNICAST_SCATTER_WRITE = 4u;

void kernel_main() {
    const uint32_t recv_l1_addr_a = get_arg(args::recv_l1_addr_a);
    const uint32_t recv_l1_addr_b = get_arg(args::recv_l1_addr_b);
    const uint32_t num_words = get_arg(args::num_words);
    const uint32_t base_value = get_arg(args::base_value);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);
    const uint32_t chunk_words_a = get_arg(args::chunk_words_a);
    const uint32_t noc_pkt_type = get_arg(args::noc_packet_type);

    volatile uint32_t* recv_a_uncached = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + recv_l1_addr_a);
    volatile uint32_t* result_uncached = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + result_l1_addr);

    while (recv_a_uncached[0] == 0) {
    }

    uint32_t pass = 1u;
    for (uint32_t i = 0; i < chunk_words_a; i++) {
        if (recv_a_uncached[i] != (base_value | i)) {
            pass = 0xDEAD0000u | i;
            break;
        }
    }

    if (pass == 1u && noc_pkt_type == HORIZON_NOC_UNICAST_SCATTER_WRITE) {
        volatile uint32_t* recv_b_uncached =
            reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + recv_l1_addr_b);
        const uint32_t chunk_words_b = num_words - chunk_words_a;
        for (uint32_t i = 0; i < chunk_words_b; i++) {
            if (recv_b_uncached[i] != (base_value | (chunk_words_a + i))) {
                pass = 0xDEAD1000u | i;
                break;
            }
        }
    }

    *result_uncached = pass;
}
