// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

void fill_page(uint32_t value, std::size_t page_offset = 0) {
    auto ptr = (get_local_cb_interface(cb_id).fifo_wr_ptr + page_offset) << 4;
    volatile tt_l1_ptr std::uint32_t* page_start = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(ptr);
    std::fill(page_start, page_start + page_size, value);
}

namespace NAMESPACE {
void MAIN {
    for (uint32_t i = 0; i < 2046; i++) {
        cb_reserve_back(cb_id, cb_step_size);
        fill_page(0xFFFF);
        cb_push_back(cb_id, cb_step_size);
    }

    DPRINT << HEX() << *get_cb_tiles_received_ptr(cb_id) + cb_step_size * page_size << ENDL();

    // Synchronize with the reader
    while (*get_cb_tiles_acked_ptr(cb_id) != *get_cb_tiles_received_ptr(cb_id)) {
    }

    // Both counter would be at 0xFFC0, 64 pages away from wrapping.
    if (*get_cb_tiles_received_ptr(cb_id) != 0xFFC0) {
        DPRINT << "Not stopping at 0xFFC0 as expected! Exiting" << ENDL();
        return;
    }

    // Here we overflow the received count and fill the buffer.
    cb_reserve_back(cb_id, cb_step_size * 2);
    fill_page(0xFF00);
    cb_push_back(cb_id, cb_step_size * 2);

    // Note: Reader is not pulling any more data out of the buffer,
    // buffer stays full.

    // Should be: Received: 0x0000, Acked: 0xFFC0
    DPRINT << "Before: Received: " << HEX() << *get_cb_tiles_received_ptr(cb_id)
           << " Acked: " << *get_cb_tiles_acked_ptr(cb_id) << ENDL();

    // This reserve should not return
    cb_reserve_back(cb_id, cb_step_size * 2);
    // This would be overwrite previous value if reserve returns prematurely.
    fill_page(0xC0FE);
    cb_push_back(cb_id, cb_step_size * 2);

    DPRINT << "Should be unreachable" << ENDL();

    DPRINT << "After: Received: " << HEX() << *get_cb_tiles_received_ptr(cb_id)
           << " Acked: " << *get_cb_tiles_acked_ptr(cb_id) << ENDL();
}
}  // namespace NAMESPACE
#else
namespace NAMESPACE {
void MAIN {}
}  // namespace NAMESPACE
#endif
