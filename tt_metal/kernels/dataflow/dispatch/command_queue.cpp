#include "tt_metal/kernels/dataflow/dispatch/command_queue.hpp"

#include "debug_print.h"

void kernel_main() {
    InterleavedAddrGen<true> dram_addr_gen;
    InterleavedAddrGen<false> l1_addr_gen;
    // Read command from host command queue... l1 read addr since
    // pulling in the actual command into l1
    static constexpr u32 command_start_addr = UNRESERVED_BASE; // Space between UNRESERVED_BASE -> data_start is for commands
    static constexpr u32 data_start_addr = DEVICE_COMMAND_DATA_ADDR;  // Hard-coded for now

    // These are totally temporary until PK checks in his changes for
    // separating kernels from firmware
    noc_prepare_deassert_reset_flag(DEASSERT_RESET_SRC_L1_ADDR);
    noc_prepare_assert_reset_flag(ASSERT_RESET_SRC_L1_ADDR);

    // Write my own NOC address to local L1 so that when I dispatch kernels,
    // they will know how to let me know they have finished
    *reinterpret_cast<volatile uint64_t*>(DISPATCH_MESSAGE_REMOTE_SENDER_ADDR) = get_noc_addr(DISPATCH_MESSAGE_ADDR);

    // For time being, while true is here until Paul's changes,
    // in which while true loop will be in the firmware

    while (true) {
        volatile u32* command_ptr = reinterpret_cast<volatile u32*>(command_start_addr);
        cq_wait_front();
        // Hardcoded for time being, need to clean this up
        uint64_t src_noc_addr = get_noc_addr(NOC_X(0), NOC_Y(4), cq_read_interface.fifo_rd_ptr << 4);

        noc_async_read(src_noc_addr, u32(command_start_addr), NUM_16B_WORDS_IN_DEVICE_COMMAND << 4);
        noc_async_read_barrier();

        // Control data
        u32 finish = command_ptr[0];              // Whether to notify the host that we have finished
        u32 num_workers = command_ptr[1];         // If num_workers > 0, it means we are launching a program
        u32 data_size_in_bytes = command_ptr[2];  // The amount of trailing data after the device command rounded to the
                                                  // nearest multiple of 32
        u32 num_buffer_reads = command_ptr[3];    // How many ReadBuffer commands we are running
        u32 num_buffer_writes = command_ptr[4];   // How many WriteBuffer commands we are running
        u32 num_program_writes =
            command_ptr[5];  // How many relays we need to make for program data (this needs more in depth explanation)

        // Will explain these magic numbers here, but soon will refactor these
        // We allocate 16 words for control information (finish, num_workers, num_buffer_reads/writes, etc)
        // We allocate 108 words since there are 108 worker cores on grayskull
        // We allocate 4 * 11 words for read/write buffers in their entirety
        // The rest is allocated for relaying program data (kernels, cbs, sem configs)
        command_ptr = reinterpret_cast<volatile u32*>(command_start_addr + (16 + 108) * sizeof(u32));
        read_buffers(num_buffer_reads, command_ptr, dram_addr_gen, l1_addr_gen);
        write_buffers(num_buffer_writes, command_ptr, dram_addr_gen, l1_addr_gen);

        command_ptr = reinterpret_cast<volatile u32*>(command_start_addr + (16 + 108 + 4 * 11) * sizeof(u32));
        write_program(num_program_writes, command_ptr);


        command_ptr = reinterpret_cast<volatile u32*>(command_start_addr + (16) * sizeof(u32));
        launch_program(num_workers, command_ptr);

        finish_program(finish);

        // This tells the dispatch core how to update its read pointer
        cq_pop_front(data_size_in_bytes + DeviceCommand::size_in_bytes());
    }
}
