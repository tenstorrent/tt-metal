/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

#include "dataflow_api.h"
#include "debug_print.h"
#include "debug_status.h"
#include "tt_metal/impl/dispatch/device_command.hpp"

template <typename T>
inline T min(T a, T b) {
    return (a < b) ? a: b;
}

template <typename T>
void write_buffer(
    T& addr_gen,
    u32 src_addr,
    u32 src_noc,
    u32 dst_addr,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size) {
    // Base address of where we are writing to
    addr_gen.bank_base_address = dst_addr;
    addr_gen.page_size = padded_page_size;

    u32 bank_id = 0;
    while (padded_buf_size > 0) {

        // Read in a big chunk of data
        u32 read_size = min(burst_size, padded_buf_size);
        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        noc_async_read(src_noc_addr, DEVICE_COMMAND_DATA_ADDR, read_size);
        padded_buf_size -= read_size;
        src_addr += read_size;
        u32 local_addr = DEVICE_COMMAND_DATA_ADDR;
        noc_async_read_barrier();

        // Send pages within the chunk to their destination
        for (u32 i = 0; i < read_size; i += padded_page_size) {
            u64 dst_addr = addr_gen.get_noc_addr(bank_id++);
            noc_async_write(local_addr, dst_addr, page_size);
            local_addr += padded_page_size;
        }
        noc_async_write_barrier();
    }
}

FORCE_INLINE void write_buffers(
    u32 num_buffer_writes,
    volatile tt_l1_ptr u32*& command_ptr,
    InterleavedAddrGen<true>& dram_addr_gen,
    InterleavedAddrGen<false>& l1_addr_gen) {
    for (u32 i = 0; i < num_buffer_writes; i++) {
        u32 src_addr = command_ptr[0];
        u32 src_noc = command_ptr[1];
        u32 dst_addr = command_ptr[2];

        u32 padded_buf_size = command_ptr[3];
        u32 burst_size = command_ptr[4];
        u32 page_size = command_ptr[5];
        u32 padded_page_size = command_ptr[6];
        u32 buf_type = command_ptr[7];

#define write_buffer_args                                                                                      \
    src_addr, src_noc, dst_addr, padded_buf_size, burst_size, page_size, padded_page_size

        u64 src_noc_addr = (u64(src_noc) << 32) | src_addr;
        switch (buf_type) {
            case 0:  // DRAM
                write_buffer(dram_addr_gen, write_buffer_args);
                break;
            case 1:  // L1
                write_buffer(l1_addr_gen, write_buffer_args);
                break;
        }

        command_ptr += 8;
    }
}

template <typename T>
FORCE_INLINE void read_buffer(
    T& addr_gen,
    u32 dst_addr,
    u32 dst_noc,
    u32 src_addr,

    u32 padded_buf_size,
    u32 burst_size,
    u32 page_size,
    u32 padded_page_size) {
    // Base address of where we are reading from
    addr_gen.bank_base_address = src_addr;
    addr_gen.page_size = padded_page_size;

    u32 bank_id = 0;
    while (padded_buf_size > 0) {
        // Read in pages until we don't have anymore memory
        // available
        u32 write_size = min(burst_size, padded_buf_size);
        u32 local_addr = DEVICE_COMMAND_DATA_ADDR;
        u64 dst_noc_addr = (u64(dst_noc) << 32) | dst_addr;
        dst_addr += write_size;
        padded_buf_size -= write_size;

        for (u32 i = 0; i < write_size; i += padded_page_size) {
            u64 src_addr = addr_gen.get_noc_addr(bank_id++);
            noc_async_read(src_addr, local_addr, page_size);
            local_addr += padded_page_size;
        }
        noc_async_read_barrier();
        noc_async_write(DEVICE_COMMAND_DATA_ADDR, dst_noc_addr, write_size);
        noc_async_write_barrier();
    }
}

FORCE_INLINE void read_buffers(
    u32 num_buffer_reads,
    volatile tt_l1_ptr u32*& command_ptr,
    InterleavedAddrGen<true>& dram_addr_gen,
    InterleavedAddrGen<false>& l1_addr_gen) {
    for (u32 i = 0; i < num_buffer_reads; i++) {
        u32 dst_addr = command_ptr[0];
        u32 dst_noc = command_ptr[1];
        u32 src_addr = command_ptr[2];

        u32 padded_buf_size = command_ptr[3];
        u32 burst_size = command_ptr[4];
        u32 page_size = command_ptr[5];
        u32 padded_page_size = command_ptr[6];
        u32 buf_type = command_ptr[7];

#define read_buffer_args                                                                                       \
    dst_addr, dst_noc, src_addr, padded_buf_size, burst_size, page_size, padded_page_size

        switch (buf_type) {
            case 0:  // DRAM
                read_buffer(dram_addr_gen, read_buffer_args);
                break;
            case 1:  // L1
                read_buffer(l1_addr_gen, read_buffer_args);
                break;
        }

        command_ptr += 8;
    }
}

FORCE_INLINE void write_program_section(
    u32 src, u32 src_noc, u32 transfer_size, u32 num_writes, volatile tt_l1_ptr u32*& command_ptr) {
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
        // DPRINT << "CHUNK" << ENDL();
        for (u32 i = 0; i < transfer_size; i += sizeof(u32)) {
            // DPRINT << *reinterpret_cast<volatile tt_l1_ptr u32*>(src + i) << ENDL();
        }
        #else
        noc_async_write_multicast(src, u64(dst_noc) << 32 | dst, transfer_size, num_receivers);
        #endif
    }
    #ifndef TT_METAL_DISPATCH_MAP_DUMP
    noc_async_write_barrier();
    #endif
}

FORCE_INLINE void write_program(u32 num_program_relays, volatile tt_l1_ptr u32*& command_ptr) {
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
        // DPRINT << "EXIT_CONDITION" << ENDL();
    }
#endif
}

FORCE_INLINE void launch_program(u32 num_workers, u32 num_multicast_messages, volatile tt_l1_ptr u32*& command_ptr) {
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
    DEBUG_STATUS('Q', 'W');
    while (reinterpret_cast<volatile tt_l1_ptr u32*>(DISPATCH_MESSAGE_ADDR)[0] != num_workers)
        ;
    DEBUG_STATUS('Q', 'D');
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

    volatile tt_l1_ptr u32* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    uint64_t finish_noc_addr = get_noc_addr(PCIE_NOC_X, PCIE_NOC_Y, HOST_CQ_FINISH_PTR);
    noc_async_write(u32(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
