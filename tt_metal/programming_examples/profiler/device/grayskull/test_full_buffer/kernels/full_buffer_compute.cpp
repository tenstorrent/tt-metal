#include <cstdint>
#include "llk_3c.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
    void MAIN {
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
} // NAMESPACE
