// attention-front bandwidth microbenchmark, BRISC (mode 3): forwards the odd blocks over NOC1 when NCRISC signals them
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    constexpr uint32_t sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t TB = 1088, PAGE_B = 10 * TB, BLK_PAGES = 8, NBLK = 14, NSLOT = 3;
    if constexpr (mode != 3) {
        return;
    }
    const uint32_t px = get_arg_val<uint32_t>(0), py = get_arg_val<uint32_t>(1);
    const uint32_t ring = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    for (uint32_t j = 1; j < NBLK; j += 2) {
        noc_semaphore_wait_min(sem, j);
        const uint32_t src = ring + (j % NSLOT) * BLK_PAGES * PAGE_B;
        noc_async_write(src, get_noc_addr(px, py, src), BLK_PAGES * PAGE_B);
    }
    noc_async_write_barrier();
}
