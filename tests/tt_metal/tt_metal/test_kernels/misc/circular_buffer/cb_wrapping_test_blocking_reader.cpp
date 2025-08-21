// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void core_agnostic_main();

#ifdef COMPILE_FOR_TRISC
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#ifdef TRISC_UNPACK
    core_agnostic_main();
#endif
}
}  // namespace NAMESPACE
#else
#include "dataflow_api.h"
void kernel_main() { core_agnostic_main(); }
#endif

#include <cstdint>

using namespace tt;

static constexpr auto CB_ID = CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

using DataT = std::uint32_t;

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

// This should be enough spining time for the writer to fill the CB with 2 steps of data.
static constexpr std::size_t NUM_WAIT_CYCLES = 1024 * 1024;

void report_page(std::size_t i) {
    invalidate_l1_cache();
    auto result_ptr = get_arg_val<DataT*>(0);
#ifdef TRISC_UNPACK
    auto read_ptr = reinterpret_cast<DataT*>(get_local_cb_interface(CB_ID).fifo_rd_ptr << 4);
    result_ptr[i] = read_ptr[0];
#elif defined(COMPILE_FOR_BRISC)
    auto read_ptr = reinterpret_cast<DataT*>(get_read_ptr(CB_ID));
    result_ptr[i] = read_ptr[0];
#endif
}

void core_agnostic_main() {
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb_wait_front(CB_ID, CB_STEP_SIZE);
        cb_pop_front(CB_ID, CB_STEP_SIZE);
    }

    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(NUM_WAIT_CYCLES);
    DPRINT << "Reader Wait Done" << ENDL();

    for (auto i = 0ul; i < 3; i++) {
        cb_wait_front(CB_ID, CB_STEP_SIZE);
        report_page(i);
        cb_pop_front(CB_ID, CB_STEP_SIZE);
    }
}
