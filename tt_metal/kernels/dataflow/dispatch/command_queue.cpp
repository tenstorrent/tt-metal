#include <stdint.h>

#include "dataflow_api.h"
#include "debug_print.h"
#include "frameworks/tt_dispatch/impl/command.hpp"

template <typename T>
void write(
    T addr_gen,
    u32 src_addr,
    u32 src_noc,
    u32 dst_addr,

    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    // Base address of where we are writing to
    addr_gen.bank_base_address = dst_addr;
    addr_gen.page_size = page_size;

    u32 id = 0;
    for (u32 j = 0; j < num_bursts; j++) {
        u32 data_addr = UNRESERVED_BASE;
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;

        noc_async_read(src_noc_addr, data_addr, burst_size);

        src_addr += burst_size;
        noc_async_read_barrier();


        for (u32 k = 0; k < num_pages_per_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_write(data_addr, addr, page_size);
            data_addr += page_size;
        }
        noc_async_write_barrier();
    }
    // In case where the final burst a different size than the others
    if (remainder_burst_size) {
        u32 data_addr = UNRESERVED_BASE;
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        noc_async_read(src_noc_addr, data_addr, remainder_burst_size);
        noc_async_read_barrier();

        for (u32 k = 0; k < num_pages_per_remainder_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_write(data_addr, addr, page_size);
            data_addr += page_size;
        }
        noc_async_write_barrier();
    }
}

template <typename T>
FORCE_INLINE void read(
    T addr_gen,
    u32 dst_addr,
    u32 dst_noc,
    u32 src_addr,

    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum) {
    // Base address of where we are reading from
    addr_gen.bank_base_address = src_addr;
    addr_gen.page_size = page_size;

    u32 id = 0;
    for (u32 j = 0; j < num_bursts; j++) {
        u32 data_addr = UNRESERVED_BASE;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;

        for (u32 k = 0; k < num_pages_per_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_read(addr, data_addr, page_size);
            data_addr += page_size;
        }
        noc_async_read_barrier();

        noc_async_write(UNRESERVED_BASE, dst_noc_addr, burst_size);
        dst_addr += burst_size;
        noc_async_write_barrier();

    }

    if (remainder_burst_size) {
        u32 data_addr = UNRESERVED_BASE;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;

        for (u32 k = 0; k < num_pages_per_remainder_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(id++);

            noc_async_read(addr, data_addr, page_size);
            data_addr += page_size;
        }
        noc_async_read_barrier();

        noc_async_write(UNRESERVED_BASE, dst_noc_addr, remainder_burst_size);
        noc_async_write_barrier();
    }
}

void kernel_main() {
    InterleavedAddrGen<true> dram_addr_gen;
    InterleavedAddrGen<false> l1_addr_gen;
    // For time being, while true is here until Paul's changes,
    // in which while true loop will be in the firmware
    while (true) {
        cq_wait_front();

        // Hardcoded for time being, need to clean this up
        uint64_t src_noc_addr = get_noc_addr(NOC_X(0), NOC_Y(4), cq_read_interface.fifo_rd_ptr << 4);

        // // Read command from host command queue... l1 read addr since
        // // pulling in the actual command into l1
        u32 command_start_addr = 150 * 1024;
        u32* command_ptr = (u32*)command_start_addr;

        // For now, hardcoding the data start, but we can definitely
        // pre-compute the right number
        noc_async_read(src_noc_addr, u32(command_start_addr), NUM_16B_WORDS_IN_COMMAND_TABLE << 4);
        noc_async_read_barrier();
        u32 finish = command_ptr[0];
        u32 launch = command_ptr[1];
        u32 data_size_in_bytes = command_ptr[2];
        u32 num_reads = command_ptr[3];
        u32 num_writes = command_ptr[4];
        command_ptr += 5;

        if (finish) {
            volatile u32* finish_ptr = get_cq_finish_ptr();
            finish_ptr[0] = 1;
            uint64_t finish_noc_addr = get_noc_addr(NOC_X(0), NOC_Y(4), HOST_CQ_FINISH_PTR);
            noc_async_write(u32(finish_ptr), finish_noc_addr, 4);
            noc_async_write_barrier();
            finish_ptr[0] = 0;
        }

        for (u32 i = 0; i < num_reads; i++) {
            u32 dst_addr = command_ptr[0];
            u32 dst_noc = command_ptr[1];
            u32 src_addr = command_ptr[2];
            u32 src_noc_start = command_ptr[3];
            u32 num_bursts = command_ptr[4];
            u32 burst_size = command_ptr[5];
            u32 num_pages_per_burst = command_ptr[6];
            u32 page_size = command_ptr[7];
            u32 remainder_burst_size = command_ptr[8];
            u32 num_pages_per_remainder_burst = command_ptr[9];
            u32 banking_enum = command_ptr[10];

            switch (banking_enum) {
                case 0: // DRAM
                    read(
                        dram_addr_gen,
                        dst_addr,
                        dst_noc,
                        src_addr,

                        num_bursts,
                        burst_size,
                        num_pages_per_burst,
                        page_size,
                        remainder_burst_size,
                        num_pages_per_remainder_burst,
                        banking_enum
                    );
                break;
                case 1: // L1
                    read(
                        l1_addr_gen,
                        dst_addr,
                        dst_noc,
                        src_addr,

                        num_bursts,
                        burst_size,
                        num_pages_per_burst,
                        page_size,
                        remainder_burst_size,
                        num_pages_per_remainder_burst,
                        banking_enum
                    );
                break;
            }
        }

        for (u32 i = 0; i < num_writes; i++) {
            u32 src_addr = command_ptr[0];
            u32 src_noc = command_ptr[1];
            u32 dst_addr = command_ptr[2];
            u32 dst_noc_start = command_ptr[3];
            u32 num_bursts = command_ptr[4];
            u32 burst_size = command_ptr[5];
            u32 num_pages_per_burst = command_ptr[6];
            u32 page_size = command_ptr[7];
            u32 remainder_burst_size = command_ptr[8];
            u32 num_pages_per_remainder_burst = command_ptr[9];
            u32 banking_enum = command_ptr[10];


            u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
            switch (banking_enum) {
                case 0:  // DRAM
                    write(
                        dram_addr_gen,
                        src_addr,
                        src_noc,
                        dst_addr,

                        num_bursts,
                        burst_size,
                        num_pages_per_burst,
                        page_size,
                        remainder_burst_size,
                        num_pages_per_remainder_burst,
                        banking_enum);
                    break;
                case 1: // L1
                    write(
                        l1_addr_gen,
                        src_addr,
                        src_noc,
                        dst_addr,

                        num_bursts,
                        burst_size,
                        num_pages_per_burst,
                        page_size,
                        remainder_burst_size,
                        num_pages_per_remainder_burst,
                        banking_enum);
                    break;
            }

            command_ptr += 7;
        }

        // This tells the dispatch core how to update its read pointer
        cq_pop_front((data_size_in_bytes >> 4) + NUM_16B_WORDS_IN_COMMAND_TABLE);
    }
}
