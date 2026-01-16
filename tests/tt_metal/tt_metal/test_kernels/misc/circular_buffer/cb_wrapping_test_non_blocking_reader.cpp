// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 *
 * Two possible invokation locations:
 * 1. Reader at UNPACKER core as Compute Kernel, no test will be performed (no cb_pages_available_at_front in compute
 *    kernel).
 * 2. Reader at a Dataflow Kernel, test will be performed.
 *
 */

void core_agnostic_main();

#ifdef COMPILE_FOR_BRISC
#include "api/dataflow/dataflow_api.h"

void kernel_main() { core_agnostic_main(); }
#else
#include "compute_kernel_api/common.h"

#include "experimental/circular_buffer.h"

// We are in compute kernel land
namespace NAMESPACE {
void MAIN { core_agnostic_main(); }
}  // namespace NAMESPACE
#endif

using namespace tt;

static constexpr auto CB_ID = CBIndex::c_0;
static constexpr std::size_t CB_STEP_SIZE = 32;

static constexpr std::size_t CHURN_TARGET = (0x10000 - 2 * CB_STEP_SIZE);
static constexpr std::size_t CHURN_LOOP_COUNT = CHURN_TARGET / CB_STEP_SIZE;

void core_agnostic_main() {
    experimental::CircularBuffer cb(CB_ID);
    for (auto i = 0ul; i < CHURN_LOOP_COUNT; i++) {
        cb.wait_front(CB_STEP_SIZE);
        cb.pop_front(CB_STEP_SIZE);
    }

#ifdef CHECK_FRONT
    auto result_ptr = get_arg_val<uint32_t*>(0);
    auto success_token = get_arg_val<uint32_t>(1);

    if (!cb.pages_available_at_front(CB_STEP_SIZE)) {
        *result_ptr = success_token;
    } else {
        *result_ptr = 0;
    }
#endif
}
