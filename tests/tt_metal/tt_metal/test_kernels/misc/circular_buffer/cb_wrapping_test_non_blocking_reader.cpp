// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

using namespace tt;

static constexpr auto CB_ID = CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

void kernel_main() {
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb_wait_front(CB_ID, CB_STEP_SIZE);
        cb_pop_front(CB_ID, CB_STEP_SIZE);
    }

    auto result_ptr = get_arg_val<uint32_t*>(0);
    *result_ptr = cb_pages_available_at_front(CB_ID, CB_STEP_SIZE);
}
