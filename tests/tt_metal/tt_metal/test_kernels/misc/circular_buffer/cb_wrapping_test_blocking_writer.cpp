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
#ifdef TRISC_PACK
    core_agnostic_main();
#endif
}
}  // namespace NAMESPACE
#else
#include "dataflow_api.h"
void kernel_main() { core_agnostic_main(); }
#endif

#include "debug/debug.h"
#include "circular_buffer.h"

using namespace tt;

/*
 * The CB holds 64 pages (2 steps).
 */

static constexpr auto CB_ID = tt::CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

using DataT = std::uint32_t;
static constexpr std::size_t PAGE_SIZE_BYTES = 16;
static constexpr std::size_t PAGE_SIZE = PAGE_SIZE_BYTES / sizeof(DataT);

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

// Values we write to the churn pages.
static constexpr DataT CHURN_LOOP_VALUE = 0xFFFF;
// Values we write to the areas that could be overwritten by incorrect reserve calls.
static constexpr DataT WRAP_WRITE_VALUE = 0xAAAA;
// Values used to overwrite the buffer in the last few pages.
static constexpr DataT WRITE_OVER_VALUE = 0xBBBB;

void fill_page(DataT value) {
#ifdef COMPILE_FOR_TRISC
    auto ptr = (get_local_cb_interface(CB_ID).fifo_wr_ptr) << cb_addr_shift;
#else
    auto ptr = get_write_ptr(CB_ID);
#endif

    auto page_start = reinterpret_cast<DataT*>(ptr);
    std::fill(page_start, page_start + PAGE_SIZE, value);
}

void fill_step(DataT value) {
    for (auto i = 0ul; i < CB_STEP_SIZE; i++) {
        fill_page(value);
    }
}

void core_agnostic_main() {
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb_reserve_back(CB_ID, CB_STEP_SIZE);
        fill_step(CHURN_LOOP_VALUE);
        cb_push_back(CB_ID, CB_STEP_SIZE);
    }

    // Synchronize with the reader
    while (*get_cb_tiles_acked_ptr(CB_ID) != *get_cb_tiles_received_ptr(CB_ID)) {
    }

    if (*get_cb_tiles_received_ptr(CB_ID) != CHURN_TARGET) {
        DPRINT << "Not stopping at churn target as expected! Got: " << HEX() << *get_cb_tiles_received_ptr(CB_ID)
               << ". Expected: " << (std::uint32_t)CHURN_TARGET << ". Exiting" << ENDL();
        return;
    }

    // We fill the buffer 1/2
    // This buffer would be overwritten if reserve returns prematurely.
    cb_reserve_back(CB_ID, CB_STEP_SIZE);
    fill_step(WRAP_WRITE_VALUE);
    cb_push_back(CB_ID, CB_STEP_SIZE);

    // Buffer should be full, also overflow the received count.
    cb_reserve_back(CB_ID, CB_STEP_SIZE);
    fill_step(WRAP_WRITE_VALUE);
    cb_push_back(CB_ID, CB_STEP_SIZE);

    // Note: Reader is not pulling any more data out of the buffer,
    // buffer stays full.

    // Acked counter should stay at CHURN_TARGET.
    auto expected_acked = CHURN_TARGET;
    if (*get_cb_tiles_acked_ptr(CB_ID) != expected_acked) {
        DPRINT << "Got: Acked: " << HEX() << *get_cb_tiles_acked_ptr(CB_ID)
               << ". Expected: " << (std::uint32_t)expected_acked << ENDL();
        return;
    }

    // This reserve should not return
    cb_reserve_back(CB_ID, CB_STEP_SIZE);
    // This would overwrite previous value if reserve returns prematurely.
    fill_step(WRITE_OVER_VALUE);
    cb_push_back(CB_ID, CB_STEP_SIZE);
}
