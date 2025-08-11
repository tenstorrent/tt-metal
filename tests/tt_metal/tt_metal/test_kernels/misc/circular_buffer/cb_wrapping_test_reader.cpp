// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer.h"
#include "debug/debug.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

using namespace tt;

static constexpr auto cb_id = tt::CBIndex::c_0;
static constexpr auto cb_step_size = 32;

void kernel_main() {
    volatile tt_l1_ptr std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(cb_id);
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(cb_id);

    for (uint32_t i = 0; i < 2046; i++) {
        cb_wait_front(cb_id, cb_step_size);
        cb_pop_front(cb_id, cb_step_size);
    }

    DPRINT << "Reader DONE" << ENDL();
}
