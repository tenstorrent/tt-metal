// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repro for #60551: the TEN-4746 workaround in reader_bmm_tile_layout_in0_sender_padding_metal2.cpp.
// Each thread does num_iter 16-byte NOC loopback reads from src_addr (this node) into a function-local
// static scratch, the same code as the matmul in0 reader. It then reports, in a 128-byte slot of
// result_addr: word 0 the scratch address, word 1 the thread id and, in the next 64-byte cache line
// (words 16..19), the scratch bytes, copied out by a second NOC read so the result shows what is in L1,
// not in the data cache.

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint32_t result_addr = get_arg(args::result_addr);
    const uint32_t num_iter = get_arg(args::num_iter);
    const uint32_t thread_id = get_my_thread_id();
    const uint32_t slot_addr = result_addr + thread_id * 128;

    Noc noc;
    UnicastEndpoint ep;

    // Same declaration and address conversion as the matmul in0 reader.
    static uint32_t ten4746_tdma_scratch[8] __attribute__((aligned(16)));
    const uint32_t ten4746_scratch_addr = (uint32_t)(uintptr_t)ten4746_tdma_scratch;
    for (uint32_t i = 0; i < num_iter; i++) {
        noc.async_read(
            ep,
            CoreLocalMem<uint32_t>(ten4746_scratch_addr),
            16,
            {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr},
            {});
        noc.async_read_barrier();
    }

    CoreLocalMem<uint32_t> slot(slot_addr);
    slot[0] = ten4746_scratch_addr;
    slot[1] = thread_id;
    flush_l2_cache_line(slot_addr);

    noc.async_read(
        ep,
        CoreLocalMem<uint32_t>(slot_addr + 64),
        16,
        {.noc_x = my_x[0], .noc_y = my_y[0], .addr = ten4746_scratch_addr},
        {});
    noc.async_read_barrier();
}
