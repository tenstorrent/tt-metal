#include <stdint.h>

#include "dataflow_api.h"
#include "debug_print.h"
#include "tt_metal/impl/dispatch/device_command.hpp"

template <typename T>
void write_buffer(
    T& addr_gen,
    u32 src_addr,
    u32 src_noc,
    u32 dst_addr,

    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum,
    u32 starting_bank_id) {
    // Base address of where we are writing to
    addr_gen.bank_base_address = dst_addr;
    addr_gen.page_size = page_size;

    u32 bank_id = starting_bank_id;
    for (u32 j = 0; j < num_bursts; j++) {
        u32 data_addr = DEVICE_COMMAND_DATA_ADDR;
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;

        noc_async_read(src_noc_addr, data_addr, burst_size);

        src_addr += burst_size;
        noc_async_read_barrier();

        for (u32 k = 0; k < num_pages_per_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(bank_id++);

            noc_async_write(data_addr, addr, page_size);
            data_addr += page_size;
        }
        noc_async_write_barrier();
    }
    // In case where the final burst size is a different size than the others
    if (remainder_burst_size) {
        u32 data_addr = DEVICE_COMMAND_DATA_ADDR;
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        noc_async_read(src_noc_addr, data_addr, remainder_burst_size);
        noc_async_read_barrier();

        for (u32 k = 0; k < num_pages_per_remainder_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(bank_id++);

            noc_async_write(data_addr, addr, page_size);
            data_addr += page_size;
        }
        noc_async_write_barrier();
    }
}

FORCE_INLINE void write_buffers(
    u32 num_buffer_writes,
    volatile u32*& command_ptr,
    InterleavedAddrGen<true>& dram_addr_gen,
    InterleavedAddrGen<false>& l1_addr_gen) {
    for (u32 i = 0; i < num_buffer_writes; i++) {
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
        u32 starting_bank_id = command_ptr[11];

#define write_buffer_args                                                                                      \
    src_addr, src_noc, dst_addr, num_bursts, burst_size, num_pages_per_burst, page_size, remainder_burst_size, \
        num_pages_per_remainder_burst, banking_enum, starting_bank_id

        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        switch (banking_enum) {
            case 0:  // DRAM
                write_buffer(dram_addr_gen, write_buffer_args);
                break;
            case 1:  // L1
                write_buffer(l1_addr_gen, write_buffer_args);
                break;
        }

        command_ptr += 12;
    }
}

template <typename T>
FORCE_INLINE void read_buffer(
    T& addr_gen,
    u32 dst_addr,
    u32 dst_noc,
    u32 src_addr,

    u32 num_bursts,
    u32 burst_size,
    u32 num_pages_per_burst,
    u32 page_size,
    u32 remainder_burst_size,
    u32 num_pages_per_remainder_burst,
    u32 banking_enum,
    u32 starting_bank_id) {
    // Base address of where we are reading from
    addr_gen.bank_base_address = src_addr;
    addr_gen.page_size = page_size;

    u32 bank_id = starting_bank_id;
    for (u32 j = 0; j < num_bursts; j++) {
        u32 data_addr = DEVICE_COMMAND_DATA_ADDR;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;

        for (u32 k = 0; k < num_pages_per_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(bank_id++);

            noc_async_read(addr, data_addr, page_size);
            data_addr += page_size;
        }
        noc_async_read_barrier();

        noc_async_write(DEVICE_COMMAND_DATA_ADDR, dst_noc_addr, burst_size);
        dst_addr += burst_size;
        noc_async_write_barrier();
    }

    if (remainder_burst_size) {
        u32 data_addr = DEVICE_COMMAND_DATA_ADDR;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;

        for (u32 k = 0; k < num_pages_per_remainder_burst; k++) {
            u64 addr = addr_gen.get_noc_addr(bank_id++);

            noc_async_read(addr, data_addr, page_size);
            data_addr += page_size;
        }
        noc_async_read_barrier();

        noc_async_write(DEVICE_COMMAND_DATA_ADDR, dst_noc_addr, remainder_burst_size);
        noc_async_write_barrier();
    }
}

FORCE_INLINE void read_buffers(
    u32 num_buffer_reads,
    volatile u32*& command_ptr,
    InterleavedAddrGen<true>& dram_addr_gen,
    InterleavedAddrGen<false>& l1_addr_gen) {
    for (u32 i = 0; i < num_buffer_reads; i++) {
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
        u32 starting_bank_id = command_ptr[11];

#define read_buffer_args                                                                                       \
    dst_addr, dst_noc, src_addr, num_bursts, burst_size, num_pages_per_burst, page_size, remainder_burst_size, \
        num_pages_per_remainder_burst, banking_enum, starting_bank_id

        switch (banking_enum) {
            case 0:  // DRAM
                read_buffer(
                    dram_addr_gen,
                    read_buffer_args);
                    break;
            case 1:  // L1
                read_buffer(
                    l1_addr_gen,
                    read_buffer_args);
                    break;
        }

        command_ptr += 12;
    }
}

FORCE_INLINE void write_program_section(
    u32 src, u32 src_noc, u32 transfer_size, u32 num_writes, volatile u32*& command_ptr) {
    // Bring in a program section into L1

    noc_async_read(((u64(src_noc) << 32) | src), DEVICE_COMMAND_DATA_ADDR, transfer_size);
    noc_async_read_barrier();

    // Write different parts of that program section to different worker cores
    for (u32 write = 0; write < num_writes; write++) {
        u32 src = command_ptr[0];

        u32 dst = command_ptr[1];
        u32 dst_noc = command_ptr[2];
        u32 transfer_size = command_ptr[3];
        u32 num_receivers = command_ptr[4];

        command_ptr += 5;

#ifdef TT_METAL_DISPATCH_MAP_DUMP
        DPRINT << "CHUNK" << ENDL();
        for (u32 i = 0; i < transfer_size; i += sizeof(u32)) {
            DPRINT << *reinterpret_cast<volatile u32*>(src + i) << ENDL();
        }
        #else
        noc_async_write_multicast(src, u64(dst_noc) << 32 | dst, transfer_size, num_receivers);
        #endif
    }
    #ifndef TT_METAL_DISPATCH_MAP_DUMP
    noc_async_write_barrier();
    #endif
}

FORCE_INLINE void write_program(u32 num_program_relays, volatile u32*& command_ptr) {
    for (u32 relay = 0; relay < num_program_relays; relay++) {
        u32 src = command_ptr[0];
        u32 src_noc = command_ptr[1];
        u32 transfer_size = command_ptr[2];
        u32 num_writes = command_ptr[3];

        command_ptr += 4;
        write_program_section(src, src_noc, transfer_size, num_writes, command_ptr);
    }

#ifdef TT_METAL_DISPATCH_MAP_DUMP
    if (num_program_relays != 0) {
        DPRINT << "EXIT_CONDITION" << ENDL();
    }
#endif
}

FORCE_INLINE void launch_program(u32 num_workers, u32 num_multicast_messages, volatile u32*& command_ptr) {
// Never launch a program when this tool is used.
#ifdef TT_METAL_DISPATCH_MAP_DUMP
    return;
#endif

    if (not num_workers)
        return;

    volatile uint32_t* message_addr_ptr = reinterpret_cast<volatile uint32_t*>(DISPATCH_MESSAGE_ADDR);
    *message_addr_ptr = 0;
    for (u32 i = 0; i < num_multicast_messages * 2; i += 2) {
        u64 worker_core_noc_coord = u64(command_ptr[i]) << 32;
        u32 num_messages = command_ptr[i + 1];
        u64 deassert_addr = worker_core_noc_coord | TENSIX_SOFT_RESET_ADDR;
        noc_semaphore_set_multicast(DEASSERT_RESET_SRC_L1_ADDR, deassert_addr, num_messages);
    }

    noc_async_write_barrier();

    // Wait on worker cores to notify me that they have completed
    while (reinterpret_cast<volatile u32*>(DISPATCH_MESSAGE_ADDR)[0] != num_workers)
        ;
    for (u32 i = 0; i < num_multicast_messages * 2; i += 2) {
        u64 worker_core_noc_coord = u64(command_ptr[i]) << 32;
        u32 num_messages = command_ptr[i + 1];
        u64 assert_addr = worker_core_noc_coord | TENSIX_SOFT_RESET_ADDR;

        noc_semaphore_set_multicast(ASSERT_RESET_SRC_L1_ADDR, assert_addr, num_messages);
    }
    noc_async_write_barrier();
}

FORCE_INLINE void finish_program(u32 finish) {
    if (not finish)
        return;

    volatile u32* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    uint64_t finish_noc_addr = get_noc_addr(PCIE_NOC_X, PCIE_NOC_Y, HOST_CQ_FINISH_PTR);
    noc_async_write(u32(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
