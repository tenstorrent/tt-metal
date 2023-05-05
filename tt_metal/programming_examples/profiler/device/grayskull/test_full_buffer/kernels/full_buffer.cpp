#include <cstdint>

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i ++)
    {
        kernel_profiler::mark_time(2);
//Max unroll size
#pragma GCC unroll 65534
        for (int j = 0 ; j < LOOP_SIZE; j++)
        {
            asm("nop");
        }
    }
}
