#include <cstdint>
#include "debug_print.h"

/**
 * add two ints
 * args are in L1
 * result is in L1
*/

void kernel_main() {

    volatile std::uint32_t* arg_a = (volatile uint32_t*)(L1_ARG_BASE);
    volatile std::uint32_t* arg_b = (volatile uint32_t*)(L1_ARG_BASE + 4);
    volatile std::uint32_t* result = (volatile uint32_t*)(L1_RESULT_BASE);

    //Sample print statement
    DPRINT << 123;
    result[0] = arg_a[0] + arg_b[0];

}
