#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    for (u32 i = 0; i < 20; i++) {
        u32 load = *reinterpret_cast<volatile tt_l1_ptr u32*>(400 * 1024);
        u32 local_load1 = *reinterpret_cast<volatile u32*>(MEM_LOCAL_BASE);
        u32 local_load2 = *reinterpret_cast<volatile u32*>(MEM_LOCAL_BASE);
        u32 local_load3 = *reinterpret_cast<volatile u32*>(MEM_LOCAL_BASE);
        u32 local_load4 = *reinterpret_cast<volatile u32*>(MEM_LOCAL_BASE);
        u32 local_load5 = *reinterpret_cast<volatile u32*>(MEM_LOCAL_BASE);
    }
}
