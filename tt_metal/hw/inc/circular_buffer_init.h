// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "circular_buffer.h"
#include "circular_buffer_constants.h"
#include "remote_circular_buffer_api.h"
#include "risc_attribs.h"

// NCRISC and BRISC setup read and write
// TRISC sets up read or write
template <bool read, bool write, bool init_wr_tile_ptr>
FORCE_INLINE void setup_local_cb_read_write_interfaces(
    uint32_t tt_l1_ptr* cb_l1_base, uint32_t start_cb_index, uint32_t local_cb_mask) {
    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr =
        cb_l1_base + start_cb_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;

    local_cb_mask >>= start_cb_index;
    uint32_t cb_id = start_cb_index;
    LocalCBInterface* local_interface_ptr = &get_local_cb_interface(cb_id);

// The following code is a C++ version of the assembly loop. It performs the same operations as the assembly, but
// the compiler doesn't optimize it as well as the assembly version (roughly 770 vs 595 cycles for 32 CBs on wormhole).
// This code is only to demonstrate the logic of the loop and is not used in production builds.
#if DISABLE_CB_ASSEMBLY

    bool next_cb_exists = local_cb_mask & 1;
    while (local_cb_mask) {
        // We could attempt to find the next set bit instead of iterating through all bits, but the circular buffers are
        // often pretty tightly packed and computing the next set bit is somewhat expensive without specialized
        // instructions.
        // TODO: Blackhole supports zbb, so use __builtin_ctz there.
        if (next_cb_exists) {
            // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
            uint32_t fifo_size = circular_buffer_config_addr[1] >> cb_addr_shift;
            uint32_t fifo_addr = circular_buffer_config_addr[0] >> cb_addr_shift;
            uint32_t fifo_num_pages = write ? circular_buffer_config_addr[2] : 0;
            uint32_t fifo_page_size = circular_buffer_config_addr[3] >> cb_addr_shift;
            uint32_t fifo_limit = fifo_addr + fifo_size;
            local_cb_mask >>= 1;
            next_cb_exists = local_cb_mask & 1;

            circular_buffer_config_addr += UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;

            LocalCBInterface& local_interface = get_local_cb_interface(cb_id);
            local_interface.fifo_limit = fifo_limit;  // to check if we need to wrap
            if (write) {
                local_interface.fifo_wr_ptr = fifo_addr;
            }
            if (read) {
                local_interface.fifo_rd_ptr = fifo_addr;
            }
            local_interface.fifo_size = fifo_size;
            local_interface.tiles_acked_received_init = 0;
            if (write) {
                local_interface.fifo_num_pages = fifo_num_pages;
            }
            local_interface.fifo_page_size = fifo_page_size;
            if (init_wr_tile_ptr) {
                local_interface.fifo_wr_tile_ptr = 0;
            }
            cb_id++;
        } else {
            circular_buffer_config_addr += UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;
            cb_id++;

            local_cb_mask >>= 1;
            next_cb_exists = local_cb_mask & 1;
        }
    }

#else

    asm volatile(
        "    j .LOOP_ENTRY%=\n\t"

        // Process a single CB.
        ".LOOP%=:\n\t"
        "    lw a3, 4(%[cbconfig])\n\t"  // fifo_size = *(circular_buffer_config_addr + 1)
        "    lw a2, 0(%[cbconfig])\n\t"  // fifo_addr = *(circular_buffer_config_addr + 0)
        ".if %[write]\n\t"
        "    lw a6, 8(%[cbconfig])\n\t"  // fifo_num_pages = *(circular_buffer_config_addr + 2)
        ".endif\n\t"
        "    lw a7, 12(%[cbconfig])\n\t"  // fifo_page_size = *(circular_buffer_config_addr + 3)

        // Fill the load latency (8 cycles) with useful work.
        "    srli %[local_cb_mask], %[local_cb_mask], 1\n\t"  // local_cb_mask >>= 1
        "    andi t0, %[local_cb_mask], 1\n\t"                // next_cb_exists = local_cb_mask & 1

        "    sw zero, %[off_tiles_acked](%[liptr])\n\t"  // local_interface.tiles_acked_received_init = 0;
        ".if %[init_wr_tile_ptr]\n\t"
        "    sw zero, %[off_fifo_tile_wr_ptr](%[liptr])\n\t"  // local_interface.fifo_wr_tile_ptr = 0;
        ".endif\n\t"

        // Advance to next cb config.
        "    addi %[cbconfig], %[cbconfig], %[circular_buffer_byte_size]\n\t"

        // 8 cycles have passed, so we can now use the loaded values.

        ".if %[cb_addr_shift] != 0\n\t"
        "    srli a3, a3, %[cb_addr_shift]\n\t"  // fifo_size >>= cb_addr_shift
        "    srli a2, a2, %[cb_addr_shift]\n\t"  // fifo_addr >>= cb_addr_shift
        "    srli a7, a7, %[cb_addr_shift]\n\t"  // fifo_page_size >>= cb_addr_shift
        ".endif\n\t"

        "    sw a3, %[off_fifo_size](%[liptr])\n\t"
        "    add a3, a2, a3\n\t"  // fifo_limit = fifo_addr + fifo_size
        "    sw a3, %[off_fifo_limit](%[liptr])\n\t"

        ".if %[write]\n\t"
        "    sw a2, %[off_fifo_wr_ptr](%[liptr])\n\t"
        "    sw a6, %[off_fifo_num_pages](%[liptr])\n\t"
        ".endif\n\t"

        ".if %[read]\n\t"
        "    sw a2, %[off_fifo_rd_ptr](%[liptr])\n\t"
        ".endif\n\t"

        "    sw a7, %[off_fifo_page_size](%[liptr])\n\t"

        "    addi %[liptr], %[liptr], %[local_cb_interface_size]\n\t"  // local_interface_ptr++;

        "    bnez t0, .LOOP%=\n\t"                     // if (next_cb_exists) goto LOOP
        "    beqz %[local_cb_mask], .LOOP_EXIT%=\n\t"  // if (local_cb_mask == 0) goto LOOP_EXIT

        // Skip over the current cb since it's not in the mask.
        ".NO_CB%=:\n\t"
        // Advance cb config address and local interface pointer
        "    addi %[cbconfig], %[cbconfig], %[circular_buffer_byte_size]\n\t"
        "    addi %[liptr], %[liptr], %[local_cb_interface_size]\n\t"

        "    srli %[local_cb_mask], %[local_cb_mask], 1\n\t"  // local_cb_mask >>= 1

        ".LOOP_ENTRY%=:\n\t"
        "    andi t0, %[local_cb_mask], 1\n\t"     // next_cb_exists = local_cb_mask & 1
        "    bnez t0, .LOOP%=\n\t"                 // if (next_cb_exists) goto LOOP
        "    bnez %[local_cb_mask], .NO_CB%=\n\t"  // if (local_cb_mask != 0) goto NO_CB

        ".LOOP_EXIT%=:\n\t"

        : [cbconfig] "+r"(circular_buffer_config_addr),
          [local_cb_mask] "+r"(local_cb_mask),
          [liptr] "+r"(local_interface_ptr)
        : [off_fifo_size] "i"(offsetof(LocalCBInterface, fifo_size)),
          [off_fifo_limit] "i"(offsetof(LocalCBInterface, fifo_limit)),
          [off_fifo_page_size] "i"(offsetof(LocalCBInterface, fifo_page_size)),
          [off_fifo_num_pages] "i"(offsetof(LocalCBInterface, fifo_num_pages)),
          [off_fifo_rd_ptr] "i"(offsetof(LocalCBInterface, fifo_rd_ptr)),
          [off_fifo_wr_ptr] "i"(offsetof(LocalCBInterface, fifo_wr_ptr)),
          [off_tiles_acked] "i"(offsetof(LocalCBInterface, tiles_acked_received_init)),
          [off_fifo_tile_wr_ptr] "i"(offsetof(LocalCBInterface, fifo_wr_tile_ptr)),
          [local_cb_interface_size] "i"(sizeof(CBInterface)),
          [circular_buffer_byte_size] "i"(UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t)),
          [read] "i"(read ? 1 : 0),
          [write] "i"(write ? 1 : 0),
          [init_wr_tile_ptr] "i"(init_wr_tile_ptr ? 1 : 0),
          [cb_addr_shift] "i"(cb_addr_shift)
        : "a2", "a3", "a6", "a7", "t0", "memory");
#endif
}

namespace experimental {

template <bool update_remote_over_noc = false>
inline void setup_remote_cb_interfaces(
    uint32_t tt_l1_ptr* cb_l1_base, uint32_t start_cb_index, uint8_t noc, uint8_t nm, bool posted, uint8_t cmd_buf) {
    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr = cb_l1_base;

    for (uint32_t cb_id = NUM_CIRCULAR_BUFFERS - 1, end_id = start_cb_index - 1; cb_id != end_id; cb_id--) {
        uint32_t config_addr = circular_buffer_config_addr[0];
        uint32_t page_size = circular_buffer_config_addr[1];
        volatile tt_l1_ptr uint32_t* l1_remote_cb_config_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_addr);
        const bool is_sender = l1_remote_cb_config_addr[0];
        uint32_t num_receivers = l1_remote_cb_config_addr[1];
        uint32_t fifo_start_addr = l1_remote_cb_config_addr[2];
        uint32_t fifo_size = l1_remote_cb_config_addr[3];
        uint32_t fifo_ptr = l1_remote_cb_config_addr[4];
        uint32_t remote_noc_xy_addr = l1_remote_cb_config_addr[5];
        uint32_t aligned_pages_sent_addr = l1_remote_cb_config_addr[6];
        if (is_sender) {
            RemoteSenderCBInterface& sender_cb_interface = get_remote_sender_cb_interface(cb_id);
            sender_cb_interface.config_ptr = config_addr;
            sender_cb_interface.fifo_start_addr = fifo_start_addr;
            sender_cb_interface.fifo_wr_ptr = fifo_ptr;
            sender_cb_interface.receiver_noc_xy_ptr = remote_noc_xy_addr;
            sender_cb_interface.aligned_pages_sent_ptr = aligned_pages_sent_addr;
            sender_cb_interface.num_receivers = num_receivers;
            // Using posted semaphore inc
            resize_remote_sender_cb_interface<update_remote_over_noc>(cb_id, page_size, noc, nm, posted, cmd_buf);
        } else {
            uint32_t aligned_pages_acked_addr = aligned_pages_sent_addr + L1_ALIGNMENT;
            uint32_t sender_noc_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr)[0];
            uint32_t sender_noc_y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr)[1];
            RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
            receiver_cb_interface.config_ptr = config_addr;
            receiver_cb_interface.fifo_start_addr = fifo_start_addr;
            receiver_cb_interface.fifo_rd_ptr = fifo_ptr;
            receiver_cb_interface.sender_noc_x = sender_noc_x;
            receiver_cb_interface.sender_noc_y = sender_noc_y;
            receiver_cb_interface.aligned_pages_acked_ptr = aligned_pages_acked_addr;
            // Using posted semaphore inc
            resize_remote_receiver_cb_interface<update_remote_over_noc>(cb_id, page_size, noc, nm, posted, cmd_buf);
        }
        circular_buffer_config_addr += UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG;
    }
}

}  // namespace experimental
