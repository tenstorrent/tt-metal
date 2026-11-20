// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric router on a Keraunos CCE0: waits for a worker's packet counter, forwards the payload to a destination
// (another HSIO tile's SRAM or a Mimir's GDDR) with an IDMA write, then returns a credit to the worker's Mimir CCE.
//
// Runtime args:
//   0: channel buffer, CCE-relative SRAM offset (packet: header word 0 = 0xFAB10000 | payload bytes)
//   1: counter, CCE-relative SRAM offset (the worker stores sequence number 1 there)
//   2: destination kind: 0 = HSIO SRAM, 2 = Mimir GDDR
//   3: destination index (HSIO tile or Mimir)   4: destination CCE (HSIO only)
//   5: destination offset
//   6: worker's global Mimir CCE   7: credit word offset in the worker's SRAM
//   8: status block, CCE-relative SRAM offset: 1 = packet seen, 2 = forwarded, 4 = done

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/keraunos_cce.h"

void kernel_main() {
    using namespace experimental::keraunos;
    const uint32_t buffer_offset = get_arg_val<uint32_t>(0);
    volatile uint32_t* counter =
        reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(get_arg_val<uint32_t>(1)));
    const uint32_t kind = get_arg_val<uint32_t>(2);
    const uint32_t index = get_arg_val<uint32_t>(3);
    const uint32_t cce = get_arg_val<uint32_t>(4);
    const uint32_t dest_offset = get_arg_val<uint32_t>(5);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(get_arg_val<uint32_t>(8)));
    volatile uint32_t* header = reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(buffer_offset));

    while (*counter != 1) {
    }
    status[1] = 1;
    const uint32_t payload_bytes = header[0] & 0xFFFF;
    const uint64_t dest = kind == 0 ? hsio_sram_addr(index, cce, dest_offset) : mimir_gddr_addr(index, dest_offset);
    async_write(local_spa_addr(my_eth_cce(), buffer_offset + 2 * sizeof(uint32_t)), dest, payload_bytes, 1);
    barrier(1);
    status[2] = 1;
    // Credit back to the worker.
    write32(mimir_cce_sram_addr(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7)), header[1]);
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
