// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer.h"
#include "debug/debug.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

using namespace tt;

static constexpr auto cb_id = tt::CBIndex::c_0;
static constexpr auto cb_step_size = 32;

static constexpr uint32_t page_size_bytes = 16;
static constexpr uint32_t page_size = page_size_bytes / sizeof(std::uint32_t);
using page_t = std::array<std::uint32_t, page_size>;

page_t read_page(std::size_t page_offset = 0) {
    std::uint32_t* read_ptr = (std::uint32_t*)get_read_ptr(cb_id) + (page_offset << 4);
    page_t result;
    std::copy(read_ptr, read_ptr + page_size, result.begin());
    return result;
}

void print_page(std::size_t page_offset = 0) {
    DPRINT << "Read Pointer: " << (int)page_offset << " " << HEX();

    for (auto element : read_page()) {
        DPRINT << HEX() << element << " ";
    }
    DPRINT << ENDL();
}

void kernel_main() {
    for (std::uint32_t i = 0; i < 2046; i++) {
        cb_wait_front(cb_id, cb_step_size);
        cb_pop_front(cb_id, cb_step_size);
    }

    // This is to give enough time for the faulty cb_reserve_back to return.
    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(1024 * 1024 * 1024);

    // If reserve_back returns prematurely, this would have been corupted.
    cb_wait_front(cb_id, cb_step_size * 2);
    print_page();
    if (read_page()[0] != 0xFF00) {
        DPRINT << "Corrupted data detected" << ENDL();
    }
    cb_pop_front(cb_id, cb_step_size * 2);

    cb_wait_front(cb_id, cb_step_size * 2);
    print_page();
    cb_pop_front(cb_id, cb_step_size * 2);
}
