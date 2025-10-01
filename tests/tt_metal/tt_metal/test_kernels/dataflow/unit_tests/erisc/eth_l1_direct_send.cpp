// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ethernet/tt_eth_api.h"
#include "ethernet/tunneling.h"
#include "debug/ring_buffer.h"
#include "ethernet/dataflow_api.h"

void kernel_main() {
    [[maybe_unused]] volatile uint32_t execution_count = 0;
    [[maybe_unused]] volatile uint32_t execution_count_2 = 0;
    // volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(0x10);
    // ptr[0] = 0;
    // ptr[1] = 0x8080;

    for (int j = 0; j < 1000; ++j) {
        // ptr[0] = execution_count++;
        // ptr[4] = execution_count_2++;
        // ptr[2] = 0xdead;
        internal_::eth_send_packet(0, 0x20000 >> 4, 0x50000 >> 4, 64 >> 4);
        // noc_async_read(0x20000, get_noc_addr(25, 25, 0x40000), 2*16*1024);
        // ptr[2] = 0xbeef;
    }

    //     for (uint32_t i = 0; i < 10000; ++i) {
    // #pragma GCC unroll 3000
    //         for (int i = 0; i < 3000; i++) {
    //             *myL1Ptr = execution_count;
    //         }
    //         execution_count++;
    //         (*myL1Ptr2)++;

    //         [[maybe_unused]] volatile uint32_t ld = *myL1Ptr;
    //     }
}
