// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common_globals.h"
#include "debug/debug.h"
#include "compute_kernel_api/common.h"
#include "circular_buffer.h"

using namespace tt;

/*
 * The CB holdes 64 pages.
 */

static constexpr auto cb_id = tt::CBIndex::c_0;
static constexpr auto cb_step_size = 32;

namespace NAMESPACE {
void MAIN {
#ifdef TRISC_PACK
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
#endif
}
}  // namespace NAMESPACE
