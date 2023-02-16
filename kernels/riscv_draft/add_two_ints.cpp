#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

/** 
 * add two ints
 * args are in L1
 * result is in L1   
*/ 

void kernel_main() {

    kernel_profiler::mark_time(1);

    volatile std::uint32_t* arg_a = (volatile uint32_t*)(L1_ARG_BASE);
    volatile std::uint32_t* arg_b = (volatile uint32_t*)(L1_ARG_BASE + 4);
    volatile std::uint32_t* result = (volatile uint32_t*)(L1_RESULT_BASE);

    result[0] = arg_a[0] + arg_b[0];

    kernel_profiler::mark_time(2);
}
    

