// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "api/compute/common.h"
#include "api/compute/experimental/semaphore.h"
#include "dev_mem_map.h"
#include "ckernel.h"

void kernel_main() {
#ifdef TRISC_MATH
    const uint32_t num_elements = get_compile_time_arg_val(0);
#if defined(INCOMING_SEM) && defined(OUTGOING_SEM)
    ckernel::Semaphore sem_in(get_compile_time_arg_val(1));
    ckernel::Semaphore sem_out(get_compile_time_arg_val(2));
#elif defined(INCOMING_SEM)
    ckernel::Semaphore sem_in(get_compile_time_arg_val(1));
#elif defined(OUTGOING_SEM)
    ckernel::Semaphore sem_out(get_compile_time_arg_val(1));
#endif

    const uint32_t buf_a = get_arg_val<uint32_t>(0);
    const uint32_t buf_b = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_elements; i++) {
#if defined(INCOMING_SEM)
        sem_in.down(1);
#endif

        const uint32_t offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        const uint32_t buf_a_addr = buf_a + MEM_L1_UNCACHED_BASE + offset;
        const uint32_t buf_b_addr = buf_b + MEM_L1_UNCACHED_BASE + offset;
        const uint32_t val = *((volatile uint32_t*)(buf_a_addr));
        const uint32_t new_val = val + 1;
        *((volatile uint32_t*)(buf_b_addr)) = new_val;

        DPRINT << "Read the value " << val << " from L1 address " << buf_a_addr << " and wrote the value " << new_val
               << " to L1 address " << buf_b_addr << ENDL();
        DEVICE_PRINT(
            "Read the value {} from L1 address {} and wrote the value {} to L1 address {}\n",
            val,
            buf_a_addr,
            new_val,
            buf_b_addr);

#if defined(OUTGOING_SEM)
        sem_out.up(1);
#endif
    }
#endif
}
