// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 *
 * Two possible invokation locations:
 * 1. Reader at UNPACKER core as Compute Kernel
 * 2. Reader at a Dataflow Kernel
 *
 * Test will be performed in both cases.
 *
 */

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
#include "api/dataflow/dataflow_api.h"
void kernel_main() { core_agnostic_main(); }
#endif

#include <cstdint>
#include "experimental/circular_buffer.h"

using namespace tt;

static constexpr auto CB_ID = CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

using DataT = std::uint32_t;
static constexpr std::size_t PAGE_SIZE_BYTES = 16;
static constexpr std::size_t PAGE_SIZE = PAGE_SIZE_BYTES / sizeof(DataT);

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

// This should be enough spining time for the writer to fill the CB with 2 steps of data.
static constexpr std::size_t NUM_WAIT_CYCLES = 1024 * 1024;

// Returns the sample of the page.
// The page should contain the same value, so the sample is the or product of every element in the page.
void report_page(std::size_t i) {
    invalidate_l1_cache();
    auto result_ptr = get_arg_val<DataT*>(0);

    DataT* read_ptr;

// Getting the raw read pointer for CB differes across TRISC and BRISC.
#ifdef TRISC_UNPACK
    read_ptr = reinterpret_cast<DataT*>(get_local_cb_interface(CB_ID).fifo_rd_ptr << cb_addr_shift);
#elif defined(COMPILE_FOR_BRISC)
    read_ptr = reinterpret_cast<DataT*>(get_read_ptr(CB_ID));
#else
    // Non unpack core on TRISC
    read_ptr = nullptr;
#endif

    if (read_ptr != nullptr) {
        result_ptr[i] = read_ptr[0];
    }
}

void core_agnostic_main() {
    experimental::CircularBuffer cb(CB_ID);
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb.wait_front(CB_STEP_SIZE);
        cb.pop_front(CB_STEP_SIZE);
    }

    DPRINT << "Reader Wait" << ENDL();
    riscv_wait(NUM_WAIT_CYCLES);
    DPRINT << "Reader Wait Done" << ENDL();

    for (auto i = 0ul; i < 3; i++) {
        cb.wait_front(CB_STEP_SIZE);
        report_page(i);
        cb.pop_front(CB_STEP_SIZE);
    }
}
