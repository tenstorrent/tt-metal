#include "api/debug/dprint.h"
#include "api/compute/common.h"
#include "api/compute/experimental/semaphore.h"
#include "dev_mem_map.h"
#include "ckernel.h"

void kernel_main() {
#ifdef TRISC_MATH
    ckernel::Semaphore sem_in(get_compile_time_arg_val(0));
    ckernel::Semaphore sem_out(get_compile_time_arg_val(1));
    const uint32_t num_elements = get_compile_time_arg_val(2);

    const uint32_t buf_a = get_arg_val<uint32_t>(0);
    const uint32_t buf_b = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_elements; i++) {
        sem_in.down(1);

        const uint32_t offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        const uint32_t buf_a_addr = buf_a + MEM_L1_UNCACHED_BASE + offset;
        const uint32_t buf_b_addr = buf_b + MEM_L1_UNCACHED_BASE + offset;
        const uint32_t val = *((volatile uint32_t*)(buf_a_addr));
        const uint32_t new_val = val + 1;
        *((volatile uint32_t*)(buf_b_addr)) = new_val;

        DPRINT << "Read the value " << val << " from L1 address " << buf_a_addr << " and wrote the value " << new_val
               << " to L1 address " << buf_b_addr << ENDL();

        sem_out.up(1);
    }
#endif
}
