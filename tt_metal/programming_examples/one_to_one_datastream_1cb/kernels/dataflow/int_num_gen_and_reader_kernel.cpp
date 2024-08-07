// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dram_buffer  = get_arg_val<uint32_t>(0);

    // NoC coords (x,y)= (1,0)
    uint64_t dram_noc_addr = get_noc_addr(1, 0, dram_buffer);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // index=0

    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    // Statistics parameters initialization
    int in_sequence = 0;
    int equal_value = 0;
    int in_order = 0;
    int out_of_order = 0;

    // Starting value
    int old_read_value = 0;
    int new_read_value = 0;

    // Write/Read data stream
    for (int integer = 1; integer < 1001; integer++) {
        // Write the integer value to the circular buffer
        uint32_t* write_ptr = (uint32_t*) l1_write_addr;
        *write_ptr = integer;

        // Write data from circular buffer to DRAM
        noc_async_write(l1_write_addr, dram_noc_addr, 4);
        noc_async_write_barrier();

        // // Busy-wait loop to introduce a delay
        // for (volatile int i = 0; i < 10000000; i++) {
        //     // Do nothing, just burn some cycles
        // }

        // Read data from DRAM to the circular buffer
        noc_async_read(dram_noc_addr, l1_write_addr, 4);
        noc_async_read_barrier();

        // Read the value from the circular buffer
        uint32_t* read_ptr = (uint32_t*) l1_write_addr;
        new_read_value = *read_ptr;

        // Debug print for read values
        //DPRINT_DATA0(DPRINT << "Iteration: " << integer << " Old Value: " << old_read_value << " New Value: " << new_read_value << "\n" << ENDL());

        // Check if values are in sequence, equal, etc.
        if (old_read_value + 1 == new_read_value) {
            in_sequence += 1;
        }
        if (old_read_value == new_read_value) {
            equal_value += 1;
        }
        if (old_read_value < new_read_value) {
            in_order += 1;
        }
        if (old_read_value > new_read_value) {
            out_of_order += 1;
        }

        old_read_value = new_read_value;
    }

    DPRINT_DATA0(DPRINT << "InSequence = " << in_sequence << "\n" << ENDL());
    DPRINT_DATA0(DPRINT << "EqualValue = " << equal_value << "\n" << ENDL());
    DPRINT_DATA0(DPRINT << "InOrder = " << in_order << "\n" << ENDL());
    DPRINT_DATA0(DPRINT << "OutOfOrder = " << out_of_order << "\n" << ENDL());
}
