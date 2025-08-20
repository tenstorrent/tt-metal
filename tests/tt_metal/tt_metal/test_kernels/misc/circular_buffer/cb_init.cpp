// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer_init.h"

#include "debug/assert.h"
#include "debug/dprint.h"

void check_cb_values(bool read, bool write, bool init_wr_tile_ptr, uint32_t initial_value, uint32_t mask) {
    for (uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        if (!(mask & (1 << i))) {
            continue;  // Skip circular buffers that are not set up
        }
        uint32_t fifo_addr = initial_value + i * 4;
        uint32_t fifo_size = initial_value + i * 4 + 1;
        uint32_t fifo_num_pages = initial_value + i * 4 + 2;
        uint32_t fifo_page_size = initial_value + i * 4 + 3;
        LocalCBInterface& local_interface = get_local_cb_interface(i);
        DPRINT << "CB " << i << " local_interface.fifo_limit: " << local_interface.fifo_limit
               << ", fifo_wr_ptr: " << local_interface.fifo_wr_ptr << ", fifo_rd_ptr: " << local_interface.fifo_rd_ptr
               << ", fifo_size: " << local_interface.fifo_size << ", fifo_num_pages: " << local_interface.fifo_num_pages
               << ", fifo_page_size: " << local_interface.fifo_page_size << ENDL();
        if (local_interface.fifo_limit != fifo_addr + fifo_size) {
            DPRINT << "FIFO limit mismatch for CB " << i << ": expected " << (fifo_addr + fifo_size) << ", got "
                   << local_interface.fifo_limit << ENDL();
            while (true);  // Purposefully hang the kernel if FIFO limit did not arrive correctly
        }
        if (write) {
            if (local_interface.fifo_wr_ptr != fifo_addr) {
                DPRINT << "FIFO write pointer mismatch for CB " << i << ": expected " << fifo_addr << ", got "
                       << local_interface.fifo_wr_ptr << ENDL();
                while (true);  // Purposefully hang the kernel if FIFO write pointer did not arrive correctly
            }
        }
        if (read) {
            if (local_interface.fifo_rd_ptr != fifo_addr) {
                DPRINT << "FIFO read pointer mismatch for CB " << i << ": expected " << fifo_addr << ", got "
                       << local_interface.fifo_rd_ptr << ENDL();
                while (true);  // Purposefully hang the kernel if FIFO read pointer did not arrive correctly
            }
        }
        if (local_interface.fifo_size != fifo_size) {
            DPRINT << "FIFO size mismatch for CB " << i << ": expected " << fifo_size << ", got "
                   << local_interface.fifo_size << ENDL();
            while (true);  // Purposefully hang the kernel if FIFO size did not arrive correctly
        }
        if (write) {
            if (local_interface.fifo_num_pages != fifo_num_pages) {
                DPRINT << "FIFO num pages mismatch for CB " << i << ": expected " << fifo_num_pages << ", got "
                       << local_interface.fifo_num_pages << ENDL();
                while (true);  // Purposefully hang the kernel if FIFO num pages did not arrive correctly
            }
        }
        if (local_interface.fifo_page_size != fifo_page_size) {
            DPRINT << "FIFO page size mismatch for CB " << i << ": expected " << fifo_page_size << ", got "
                   << local_interface.fifo_page_size << ENDL();
            while (true);  // Purposefully hang the kernel if FIFO page size did not arrive correctly
        }
        if (local_interface.tiles_acked_received_init != 0) {
            DPRINT << "Tiles acked received init mismatch for CB " << i << ": expected 0, got "
                   << local_interface.tiles_acked_received_init << ENDL();
            while (true);  // Purposefully hang the kernel if tiles acked received init did not arrive correctly
        }
        if (init_wr_tile_ptr) {
            if (local_interface.fifo_wr_tile_ptr != 0) {
                DPRINT << "FIFO write tile pointer mismatch for CB " << i << ": expected 0, got "
                       << local_interface.fifo_wr_tile_ptr << ENDL();
                while (true);  // Purposefully hang the kernel if FIFO write tile pointer did not arrive correctly
            }
        }
    }
}

uint32_t get_clock_lo() {
    volatile uint tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return *clock_lo;
}

template <bool read, bool write, bool init_wr_tile_ptr>
void perform_test(uint32_t initial_value) {
    uint32_t mask = get_arg_val<uint32_t>(0);
    DPRINT << "Performing test with read: " << static_cast<uint32_t>(read)
           << ", write: " << static_cast<uint32_t>(write)
           << ", init_wr_tile_ptr: " << static_cast<uint32_t>(init_wr_tile_ptr) << ", mask: " << mask << ENDL();

    uint32_t tt_l1_ptr* cb_l1_base = get_arg_val<uint32_t tt_l1_ptr*>(1);

    for (uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS * 4; i++) {
        ((volatile uint32_t*)cb_l1_base)[i] = initial_value + i;
    }
    uint8_t* out_data = reinterpret_cast<uint8_t*>(cb_interface);
    for (uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS * sizeof(CBInterface); i++) {
        out_data[i] = 0xff;
    }

    uint32_t start_time = get_clock_lo();
    setup_local_cb_read_write_interfaces<read, write, init_wr_tile_ptr>(cb_l1_base, 0, mask);
    uint32_t end_time = get_clock_lo();
    DPRINT << "Time taken for setup: " << (end_time - start_time) << " cycles" << ENDL();

    check_cb_values(read, write, init_wr_tile_ptr, initial_value, mask);
}

void kernel_main() {
    perform_test<true, true, false>(0);
    perform_test<true, false, true>(1000);
    perform_test<false, true, true>(2000);
    perform_test<false, false, true>(3000);
}
