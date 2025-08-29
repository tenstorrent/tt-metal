// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 *
 * Two possible invokation locations:
 * 1. Writer at Packer core as Compute Kernel, no test will be performed (no cb_pages_reservable_at_back in compute
 *    kernel).
 * 2. Writer at a Dataflow Kernel, test will be performed.
 *
 */

void core_agnostic_main();

#ifdef COMPILE_FOR_TRISC
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN { core_agnostic_main(); }
}  // namespace NAMESPACE
#else
#include "dataflow_api.h"
void kernel_main() { core_agnostic_main(); }
#endif

using namespace tt;

static constexpr auto CB_ID = CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

void core_agnostic_main() {
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb_reserve_back(CB_ID, CB_STEP_SIZE);
        cb_push_back(CB_ID, CB_STEP_SIZE);
    }

#ifdef CHECK_BACK
    // We fill the buffer.
    cb_reserve_back(CB_ID, CB_STEP_SIZE);
    cb_push_back(CB_ID, CB_STEP_SIZE);
    cb_reserve_back(CB_ID, CB_STEP_SIZE);
    cb_push_back(CB_ID, CB_STEP_SIZE);

    auto result_ptr = get_arg_val<uint32_t*>(0);
    auto success_token = get_arg_val<uint32_t>(1);

    if (!cb_pages_reservable_at_back(CB_ID, CB_STEP_SIZE)) {
        *result_ptr = success_token;
    } else {
        *result_ptr = 0;
    }
#endif
}
