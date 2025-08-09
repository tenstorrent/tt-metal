// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <tuple>
#include "circular_buffer.h"
#include "debug/debug.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

using namespace tt;

/*
 * The CB holdes 64 pages.
 */

static constexpr auto cb_id = tt::CBIndex::c_0;
static constexpr auto cb_step_size = 32;

void kernel_main() {
    // We bring the acked and received a single page (32) from overflow
    for (uint32_t i = 0; i < 2046; i++) {
        cb_reserve_back(cb_id, cb_step_size);
        cb_push_back(cb_id, cb_step_size);
    }

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
    cb_push_back(cb_id, cb_step_size * 2);

    // Note: Reader is not pulling any more data out of the buffer,
    // buffer stays full.

    // Should be: Received: 0x0000, Acked: 0xFFC0
    DPRINT << "Before: Received: " << HEX() << *get_cb_tiles_received_ptr(cb_id)
           << " Acked: " << *get_cb_tiles_acked_ptr(cb_id) << ENDL();

    // This reserve should not return
    cb_reserve_back(cb_id, 1);

    DPRINT << "Should be unreachable" << ENDL();

    DPRINT << "After: Received: " << HEX() << *get_cb_tiles_received_ptr(cb_id)
           << " Acked: " << *get_cb_tiles_acked_ptr(cb_id) << ENDL();
}
