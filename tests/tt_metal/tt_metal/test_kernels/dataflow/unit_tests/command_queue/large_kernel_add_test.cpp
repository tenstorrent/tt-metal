// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Default to 50000 adds
#ifndef NUM_ADDS
#define NUM_ADDS 50000
#endif

void kernel_main() {
    uint32_t result_addr = get_arg_val<uint32_t>(0);

    volatile uint32_t counter = 0;

    // Use inline assembly to generate exactly NUM_ADDS add instructions
    // This uses the .rept assembler directive which repeats the instruction
    asm volatile(
        "li t1, 0\n"        // Initialize t1 = 0
        ".rept %1\n"        // Repeat NUM_ADDS times
        "addi t1, t1, 1\n"  // Add 1 to t1 (generates NUM_ADDS add instructions)
        ".endr\n"           // End repeat
        "mv %0, t1\n"       // Move result to counter
        : "=r"(counter)     // Output: counter
        : "i"(NUM_ADDS)     // Input: NUM_ADDS
        : "t1");

    // Write the result to L1 for host to verify
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
    *result = counter;
}
