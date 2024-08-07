// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT

// Be sure to 'export TT_METAL_DPRINT_CORES=5,0' before running kernel

void kernel_main() {
    uint32_t dram_buffer  = get_arg_val<uint32_t>(0);

    // NoC coords (x,y)= (1,0)
    uint64_t dram_noc_addr = get_noc_addr(1, 0, dram_buffer);

    constexpr uint32_t cb_write = tt::CB::c_in0; // index=0
    constexpr uint32_t cb_read = tt::CB::c_in1; // index=1

    uint32_t l1_write_addr = get_write_ptr(cb_write);
    uint32_t l1_read_addr = get_write_ptr(cb_read);

    // Function to reset statistics
    auto reset_stats = [](int &in_sequence, int &equal_value, int &in_order, int &out_of_order, int &old_read_value, int &new_read_value, uint32_t* &write_ptr, uint32_t* &read_ptr) {
        in_sequence = 0;
        equal_value = 0;
        in_order = 0;
        out_of_order = 0;
        old_read_value = 0;
        new_read_value = 0;
        *write_ptr = 0;
        *read_ptr = 0;
    };

    // Function to print results in table format
    auto print_results = [](const char* config, int in_sequence, int equal_value, int in_order, int out_of_order) {
        DPRINT_DATA0(DPRINT << "| " << config << " | "
                            << in_sequence << " | "
                            << equal_value << " | "
                            << in_order << " | "
                            << out_of_order << " |\n" << ENDL());
    };

    // Header for the table
    DPRINT_DATA0(DPRINT << "-------------------------------------------------\n" << ENDL());
    DPRINT_DATA0(DPRINT << "| Configuration | InSequence | EqualValue | InOrder | OutOfOrder |\n" << ENDL());
    DPRINT_DATA0(DPRINT << "-------------------------------------------------\n" << ENDL());

    // Statistics parameters initialization
    int in_sequence = 0;
    int equal_value = 0;
    int in_order = 0;
    int out_of_order = 0;
    int old_read_value = 0;
    int new_read_value = 0;
    uint32_t *write_ptr = 0;
    uint32_t *read_ptr = 0;

    // No Barriers
    for (int integer = 1; integer < 1001; integer++) {
        write_ptr = (uint32_t*) l1_write_addr;
        *write_ptr = integer;
        noc_async_write(l1_write_addr, dram_noc_addr, 4);
        noc_async_read(dram_noc_addr, l1_read_addr, 4);
        read_ptr = (uint32_t*) l1_read_addr;
        new_read_value = *read_ptr;

        if (old_read_value + 1 == new_read_value) {
            in_sequence += 1;
        }
        if (old_read_value == new_read_value) {
            equal_value += 1;
            //DPRINT_DATA0(DPRINT << "Equal Value (No Barriers): " << new_read_value << "\n" << ENDL());
        }
        if (old_read_value < new_read_value) {
            in_order += 1;
        }
        if (old_read_value > new_read_value) {
            out_of_order += 1;
            //DPRINT_DATA0(DPRINT << "Out of Order (No Barriers): Old Value: " << old_read_value << ", New Value: " << new_read_value << "\n" << ENDL());
        }

        old_read_value = new_read_value;
    }
    print_results("No Barriers   ", in_sequence, equal_value, in_order, out_of_order);

    // Reset statistics
    reset_stats(in_sequence, equal_value, in_order, out_of_order, old_read_value, new_read_value, write_ptr, read_ptr);

    // Read Barrier
    for (int integer = 1; integer < 1001; integer++) {
        write_ptr = (uint32_t*) l1_write_addr;
        *write_ptr = integer;
        noc_async_write(l1_write_addr, dram_noc_addr, 4);
        noc_async_read(dram_noc_addr, l1_read_addr, 4);
        noc_async_read_barrier();
        read_ptr = (uint32_t*) l1_read_addr;
        new_read_value = *read_ptr;

        if (old_read_value + 1 == new_read_value) {
            in_sequence += 1;
        }
        if (old_read_value == new_read_value) {
            equal_value += 1;
            //DPRINT_DATA0(DPRINT << "Equal Value (Read Barrier): " << new_read_value << "\n" << ENDL());
        }
        if (old_read_value < new_read_value) {
            in_order += 1;
        }
        if (old_read_value > new_read_value) {
            out_of_order += 1;
            //DPRINT_DATA0(DPRINT << "Out of Order (Read Barrier): Old Value: " << old_read_value << ", New Value: " << new_read_value << "\n" << ENDL());
        }

        old_read_value = new_read_value;
    }
    print_results("Read Barrier  ", in_sequence, equal_value, in_order, out_of_order);

    // Reset statistics
    reset_stats(in_sequence, equal_value, in_order, out_of_order, old_read_value, new_read_value, write_ptr, read_ptr);

    // Write Barrier
    for (int integer = 1; integer < 1001; integer++) {
        write_ptr = (uint32_t*) l1_write_addr;
        *write_ptr = integer;
        noc_async_write(l1_write_addr, dram_noc_addr, 4);
        noc_async_write_barrier();
        noc_async_read(dram_noc_addr, l1_read_addr, 4);
        read_ptr = (uint32_t*) l1_read_addr;
        new_read_value = *read_ptr;

        if (old_read_value + 1 == new_read_value) {
            in_sequence += 1;
        }
        if (old_read_value == new_read_value) {
            equal_value += 1;
            //DPRINT_DATA0(DPRINT << "Equal Value (Write Barrier): " << new_read_value << "\n" << ENDL());
        }
        if (old_read_value < new_read_value) {
            in_order += 1;
        }
        if (old_read_value > new_read_value) {
            out_of_order += 1;
            //DPRINT_DATA0(DPRINT << "Out of Order (Write Barrier): Old Value: " << old_read_value << ", New Value: " << new_read_value << "\n" << ENDL());
        }

        old_read_value = new_read_value;
    }
    print_results("Write Barrier ", in_sequence, equal_value, in_order, out_of_order);

    // Reset statistics
    reset_stats(in_sequence, equal_value, in_order, out_of_order, old_read_value, new_read_value, write_ptr, read_ptr);

    // Both Barriers
    for (int integer = 1; integer < 1001; integer++) {
        write_ptr = (uint32_t*) l1_write_addr;
        *write_ptr = integer;
        noc_async_write(l1_write_addr, dram_noc_addr, 4);
        noc_async_write_barrier();
        noc_async_read(dram_noc_addr, l1_read_addr, 4);
        noc_async_read_barrier();
        read_ptr = (uint32_t*) l1_read_addr;
        new_read_value = *read_ptr;

        if (old_read_value + 1 == new_read_value) {
            in_sequence += 1;
        }
        if (old_read_value == new_read_value) {
            equal_value += 1;
            //DPRINT_DATA0(DPRINT << "Equal Value (Both Barriers): " << new_read_value << "\n" << ENDL());
        }
        if (old_read_value < new_read_value) {
            in_order += 1;
        }
        if (old_read_value > new_read_value) {
            out_of_order += 1;
            //DPRINT_DATA0(DPRINT << "Out of Order (Both Barriers): Old Value: " << old_read_value << ", New Value: " << new_read_value << "\n" << ENDL());
        }

        old_read_value = new_read_value;
    }
    print_results("Both Barriers ", in_sequence, equal_value, in_order, out_of_order);

    // Footer for the table
    DPRINT_DATA0(DPRINT << "-------------------------------------------------\n" << ENDL());
}
