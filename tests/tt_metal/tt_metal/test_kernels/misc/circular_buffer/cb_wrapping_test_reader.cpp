// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
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

static constexpr uint32_t CHURN_LOOP_COUNT = 0xFFC0 / cb_step_size;

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

void report_result(uint32_t page0, uint32_t page1) {
    auto result_ptr = get_arg_val<uint32_t>(0);
    DPRINT << "Result Buffer: " << HEX() << result_ptr << ENDL();
    auto result_buffer = (volatile uint32_t*)result_ptr;

    DPRINT << "Reporting page0: " << HEX() << page0 << " page1: " << HEX() << page1 << ENDL();

    result_buffer[0] = page0;
    DPRINT << "Result Buffer [0]: " << result_buffer[0] << ENDL();

    result_buffer[1] = page1;
    DPRINT << "Result Buffer [1]: " << result_buffer[1] << ENDL();
}

void kernel_main() {
    for (std::uint32_t i = 0; i < CHURN_LOOP_COUNT; i++) {
        cb_wait_front(cb_id, cb_step_size);
        cb_pop_front(cb_id, cb_step_size);
    }

    // This is to give enough time for the faulty cb_reserve_back to return.
    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(1024 * 1024 * 1024);
    DPRINT << "Reader Wait Done" << ENDL();

    cb_wait_front(cb_id, cb_step_size);
    DPRINT << "Should be 0xFFFF" << ENDL();
    print_page();
    cb_pop_front(cb_id, cb_step_size);

    // This is where the counter is wrapped around.
    // If reserve_back returns prematurely, this would have been corupted.
    cb_wait_front(cb_id, cb_step_size);
    DPRINT << "Should be 0xFF00" << ENDL();
    print_page();
    if (auto result = read_page()[0]; result != 0xFF00) {
        DPRINT << "Corrupted data detected, got: " << HEX() << result << ENDL();
    }
    auto page0 = read_page()[0];
    cb_pop_front(cb_id, cb_step_size);

    cb_wait_front(cb_id, cb_step_size);
    DPRINT << "Should be 0xC0FE" << ENDL();
    print_page();
    auto page1 = read_page()[0];
    cb_pop_front(cb_id, cb_step_size);

    report_result(page0, page1);
}
