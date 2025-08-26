// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "circular_buffer.h"
#include "debug/debug.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

using namespace tt;

static constexpr auto CB_ID = tt::CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

using DataT = std::uint32_t;

static constexpr std::size_t PAGE_SIZE_BYTES = 16;
static constexpr std::size_t PAGE_SIZE = PAGE_SIZE_BYTES / sizeof(DataT);

// A span would be more appropriate here, but we are currently in C++17 land.
using page_t = std::array<DataT, PAGE_SIZE>;

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

// This should be enough spining time for the writer to fill the CB with 2 steps of data.
static constexpr std::size_t NUM_WAIT_CYCLES = 1024 * 1024;

page_t read_page() {
    auto read_ptr = reinterpret_cast<DataT*>(get_read_ptr(CB_ID));
    page_t result;
    std::copy(read_ptr, read_ptr + PAGE_SIZE, result.begin());
    return result;
}

void kernel_main() {
    auto result_ptr = get_arg_val<DataT*>(0);

    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb_wait_front(CB_ID, CB_STEP_SIZE);
        cb_pop_front(CB_ID, CB_STEP_SIZE);
    }

    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(NUM_WAIT_CYCLES);
    DPRINT << "Reader Wait Done" << ENDL();

    for (auto i = 0ul; i < 3; i++) {
        cb_wait_front(CB_ID, CB_STEP_SIZE);
        invalidate_l1_cache();
        result_ptr[i] = read_page()[0];
        cb_pop_front(CB_ID, CB_STEP_SIZE);
    }
}
