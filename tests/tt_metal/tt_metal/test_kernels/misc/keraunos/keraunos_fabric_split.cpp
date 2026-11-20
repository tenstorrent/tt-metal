// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender/receiver split of one HSIO tile, mirroring Blackhole's erisc0/erisc1: CCE0 owns a channel buffer in its SRAM
// and tells CCE1 a packet is ready with peer_signal; CCE1 forwards the payload with an IDMA write and returns the
// credit with peer_signal. Flags live in CCE0's SRAM.
//
// Runtime args:
//   0: role: 0 = sender (CCE0), 1 = receiver (CCE1)
//   1: channel buffer, CCE-relative offset in CCE0's SRAM; word 0 = payload bytes, payload from word 1
//   2: ready flag offset   3: credit flag offset
//   4: destination kind: 0 = HSIO SRAM, 2 = Mimir GDDR   5: destination index   6: destination CCE
//   7: destination offset
//   8: status block, own SRAM offset: 1 = signalled / forwarded, 4 = done

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/keraunos_cce.h"

void kernel_main() {
    using namespace experimental::keraunos;
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t buffer_offset = get_arg_val<uint32_t>(1);
    const uint32_t ready = get_arg_val<uint32_t>(2);
    const uint32_t credit = get_arg_val<uint32_t>(3);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(get_arg_val<uint32_t>(8)));
    if (role == 0) {
        peer_signal(ready, 1);
        status[1] = 1;
        peer_wait(credit, 1);
    } else {
        peer_wait(ready, 1);
        const volatile uint32_t* header =
            reinterpret_cast<const volatile uint32_t*>(eth_cce_sram_uncached_addr(0, buffer_offset));
        const uint32_t payload_bytes = header[0];
        const uint32_t kind = get_arg_val<uint32_t>(4);
        const uint64_t dest = kind == 0 ? hsio_sram_addr(get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7))
                                        : mimir_gddr_addr(get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(7));
        async_write(local_spa_addr(0, buffer_offset + sizeof(uint32_t)), dest, payload_bytes, 1);
        barrier(1);
        status[1] = 1;
        peer_signal(credit, 1);
    }
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
