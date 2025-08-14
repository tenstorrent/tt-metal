// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

void kernel_main() {
    auto result_ptr = get_arg_val<std::uint32_t*>(0);

    for (std::uint32_t i = 0; i < CHURN_LOOP_COUNT; i++) {
        cb_wait_front(cb_id, cb_step_size);
        cb_pop_front(cb_id, cb_step_size);
    }

    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(1024 * 1024 * 1024);
    DPRINT << "Reader Wait Done" << ENDL();

    for (auto i = 0ul; i < 3; i++) {
        cb_wait_front(cb_id, cb_step_size);
        result_ptr[i] = read_page()[0];
        cb_pop_front(cb_id, cb_step_size);
    }
}
