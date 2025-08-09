// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/debug.h"
#include "compute_kernel_api/common.h"
#include "circular_buffer.h"

using namespace tt;

static constexpr auto cb_id = tt::CBIndex::c_0;
static constexpr auto cb_step_size = 32;

namespace NAMESPACE {
void MAIN {
    volatile tt_l1_ptr std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(cb_id);
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(cb_id);

    for (uint32_t i = 0; i < 2046; i++) {
        cb_wait_front(cb_id, cb_step_size);
        cb_pop_front(cb_id, cb_step_size);
    }

    DPRINT << "Reader DONE" << ENDL();
}
}  // namespace NAMESPACE
