// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repro for #63895, device side. For each offset into NEO cluster semaphore 16's window, measures how much one 4-byte
// read changes the count, once through a NoC read of this core's own register and once through a plain load. Each
// result is {value the NoC read returned, count change from the NoC read, count change from the load}.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t sem_idx = 16;
constexpr uint32_t init_value = 100;
constexpr uint32_t offsets[] = {0x0C, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34};

volatile uint32_t* sem_word(uint32_t offset) {
    return reinterpret_cast<volatile uint32_t*>(TENSIX_GLOBAL_SEM_BASE + sem_idx * TENSIX_GLOBAL_SEM_STRIDE + offset);
}

// A read at +0x20 is POST 0: it returns the count without changing it before the return.
int32_t count_change_since_init() { return static_cast<int32_t>(*sem_word(0x20)) - static_cast<int32_t>(init_value); }

}  // namespace

void kernel_main() {
    constexpr uint32_t scratch_addr = get_arg(args::scratch_addr);  // 64 B aligned
    constexpr uint32_t results_addr = get_arg(args::results_addr);

    Noc noc;
    UnicastEndpoint self;
    auto* results = reinterpret_cast<volatile uint32_t*>(results_addr + MEM_L1_UNCACHED_BASE);

    for (uint32_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        const uint32_t offset = offsets[i];

        // Destination shares the source's alignment within a 64 B line.
        const uint32_t dst = scratch_addr + offset;
        tensix_global_sem_init(sem_idx, init_value);
        CoreLocalMem<uint32_t> dst_mem(dst);
        noc.async_read(
            self,
            dst_mem,
            sizeof(uint32_t),
            {.noc_x = my_x[noc.get_noc_id()],
             .noc_y = my_y[noc.get_noc_id()],
             .addr = static_cast<uint32_t>(TENSIX_GLOBAL_SEM_BASE + sem_idx * TENSIX_GLOBAL_SEM_STRIDE + offset)},
            {});
        noc.async_read_barrier();
        results[i * 3 + 0] = *reinterpret_cast<volatile uint32_t*>(dst + MEM_L1_UNCACHED_BASE);
        results[i * 3 + 1] = static_cast<uint32_t>(count_change_since_init());

        tensix_global_sem_init(sem_idx, init_value);
        (void)*sem_word(offset);
        results[i * 3 + 2] = static_cast<uint32_t>(count_change_since_init());
    }
}
