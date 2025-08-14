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

// A span would be more appropriate here, but we are currently in C++17 land.
using page_t = std::array<std::uint32_t, page_size>;

static constexpr uint32_t CHURN_TARGET = (0x10000 - 2 * cb_step_size);
static constexpr uint32_t CHURN_LOOP_COUNT = CHURN_TARGET / cb_step_size;

// This should be enough spining time for the writer to fill the CB with 2 steps of data.
static constexpr uint32_t NUM_WAIT_CYCLES = 1024 * 1024 * 1024;

page_t read_page() {
    std::uint32_t* read_ptr = (std::uint32_t*)get_read_ptr(cb_id);
    page_t result;
    std::copy(read_ptr, read_ptr + page_size, result.begin());
    return result;
}

void kernel_main() {
    auto result_ptr = get_arg_val<std::uint32_t*>(0);

    for (std::uint32_t i = 0; i < CHURN_LOOP_COUNT; i++) {
        cb_wait_front(cb_id, cb_step_size);
        cb_pop_front(cb_id, cb_step_size);
    }

    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(NUM_WAIT_CYCLES);
    DPRINT << "Reader Wait Done" << ENDL();

    for (auto i = 0ul; i < 3; i++) {
        cb_wait_front(cb_id, cb_step_size);
        result_ptr[i] = read_page()[0];
        cb_pop_front(cb_id, cb_step_size);
    }
}
