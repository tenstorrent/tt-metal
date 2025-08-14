// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"

#ifdef TRISC_PACK

#include <algorithm>
#include "debug/debug.h"
#include "circular_buffer.h"

using namespace tt;

/*
 * The CB holdes 64 pages.
 */

static constexpr auto cb_id = tt::CBIndex::c_0;
static constexpr auto cb_step_size = 32;

static constexpr uint32_t page_size_bytes = 16;
static constexpr uint32_t page_size = page_size_bytes / sizeof(std::uint32_t);

static constexpr uint32_t CHURN_TARGET = (0x10000 - 2 * cb_step_size);
static constexpr uint32_t CHURN_LOOP_COUNT = CHURN_TARGET / cb_step_size;

// Values we write to the churn pages.
static constexpr uint32_t CHURN_LOOP_VALUE = 0xFFFF;
// Values we write to the areas that could be overwritten by incorrect reserve calls.
static constexpr uint32_t WRAP_WRITE_VALUE = 0xAAAA;
// Values used to overwrite the buffer in the last few pages.
static constexpr uint32_t WRITE_OVER_VALUE = 0xBBBB;

void fill_page(uint32_t value, std::size_t page_offset = 0) {
    auto ptr = (get_local_cb_interface(cb_id).fifo_wr_ptr + page_offset) << 4;
    volatile tt_l1_ptr std::uint32_t* page_start = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(ptr);
    std::fill(page_start, page_start + page_size, value);
}

void fill_step(uint32_t value, std::size_t step_offset = 0) {
    for (std::size_t i = 0; i < cb_step_size; i++) {
        fill_page(value, step_offset * cb_step_size + i);
    }
}

namespace NAMESPACE {
void MAIN {
    for (uint32_t i = 0; i < CHURN_LOOP_COUNT; i++) {
        cb_reserve_back(cb_id, cb_step_size);
        fill_step(CHURN_LOOP_VALUE);
        cb_push_back(cb_id, cb_step_size);
    }

    // Synchronize with the reader
    while (*get_cb_tiles_acked_ptr(cb_id) != *get_cb_tiles_received_ptr(cb_id)) {
    }

    if (*get_cb_tiles_received_ptr(cb_id) != CHURN_TARGET) {
        DPRINT << "Not stopping at " << HEX() << CHURN_TARGET << " as expected! Exiting" << ENDL();
        return;
    }

    // We fill the buffer 1/2
    // This buffer would be overwritten if reserve returns prematurely.
    cb_reserve_back(cb_id, cb_step_size);
    fill_step(WRAP_WRITE_VALUE);
    cb_push_back(cb_id, cb_step_size);

    // Buffer should be full, also overflow the received count.
    cb_reserve_back(cb_id, cb_step_size);
    fill_step(WRAP_WRITE_VALUE);
    cb_push_back(cb_id, cb_step_size);

    // Note: Reader is not pulling any more data out of the buffer,
    // buffer stays full.

    // // Should be: Received: 0x0000, Acked: 0xFFC0
    // DPRINT << "Before cb_reserve_back" << ENDL();
    // DPRINT << "Expected: Received: 0x0000, Acked: 0xFFC0" << ENDL();
    // DPRINT << "Got: Received: " << HEX() << *get_cb_tiles_received_ptr(cb_id)
    //        << " Acked: " << *get_cb_tiles_acked_ptr(cb_id) << ENDL();

    // This reserve should not return
    cb_reserve_back(cb_id, cb_step_size);
    // This would be overwrite previous value if reserve returns prematurely.
    fill_step(WRITE_OVER_VALUE);
    cb_push_back(cb_id, cb_step_size);

    // DPRINT << "Should be unreachable" << ENDL();
    // DPRINT << "After: Received: " << HEX() << *get_cb_tiles_received_ptr(cb_id)
    //        << " Acked: " << *get_cb_tiles_acked_ptr(cb_id) << ENDL();
}
}  // namespace NAMESPACE
#else
namespace NAMESPACE {
void MAIN {}
}  // namespace NAMESPACE
#endif
