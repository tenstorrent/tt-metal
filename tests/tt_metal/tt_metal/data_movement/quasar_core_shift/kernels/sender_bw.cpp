// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// L1-to-L1 NOC unicast write kernel for Quasar.
// Sends num_transactions writes of bytes_per_transaction from local L1 to a
// remote core's L1, then barriers. Times the transfer with the RISC-V cycle CSR
// and stores the elapsed cycle count to L1 for the host to read.
//
// NOTE: the wall-clock debug register read via get_timestamp() crashes the
// simulator, so we use the RISC-V `rdcycle` CSR instead.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// Read the RISC-V cycle CSR. The "memory" clobber acts as a compiler barrier so
// the surrounding NOC writes cannot be reordered outside the timed region.
inline uint32_t rdcycle() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c)::"memory");
    return c;
}

void kernel_main() {
    uint32_t data_addr = get_arg_val<uint32_t>(0);
    uint32_t cycles_addr = get_arg_val<uint32_t>(1);
    uint32_t num_transactions = get_arg_val<uint32_t>(2);
    uint32_t bytes_per_transaction = get_arg_val<uint32_t>(3);
    uint32_t receiver_noc_x = get_arg_val<uint32_t>(4);
    uint32_t receiver_noc_y = get_arg_val<uint32_t>(5);

    const uint64_t dst_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, data_addr);

    DPRINT << "sender_bw start dst_noc=(" << receiver_noc_x << "," << receiver_noc_y << ")"
           << " txns=" << num_transactions << " size=" << bytes_per_transaction << ENDL();

    uint32_t start_cycle = rdcycle();

    for (uint32_t i = 0; i < num_transactions; ++i) {
        noc_async_write(data_addr, dst_noc_addr, bytes_per_transaction);
    }
    noc_async_write_barrier();

    uint32_t end_cycle = rdcycle();

    uint32_t cycles = end_cycle - start_cycle;  // modular subtraction handles 32-bit wrap
    // Write via the uncached L1 alias so the CPU store bypasses the data cache and
    // is immediately visible to the host's L1 read-back (the NOC data writes already
    // bypass the cache, which is why correctness passes but a plain cached store of
    // the cycle count read back as 0).
    volatile tt_l1_ptr uint32_t* cycles_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cycles_addr + MEM_L1_UNCACHED_BASE);
    cycles_ptr[0] = cycles;
    cycles_ptr[1] = 0;  // host reads a 64-bit slot; clear the high word (rdcycle is 32-bit)

    DPRINT << "sender_bw done txns=" << num_transactions << " size=" << bytes_per_transaction << " cycles=" << cycles
           << ENDL();
}
