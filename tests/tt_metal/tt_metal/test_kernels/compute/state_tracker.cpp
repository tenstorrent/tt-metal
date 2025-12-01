// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define TT_METAL_STATE_TRACKER_TESTING_ENABLED

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/assert.h"  // Required in all kernels using watcher asserts
#include "debug/waypoint.h"
#include "tt_metal/include/compute_kernel_api/state_tracker.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(1);

    SET_CALLED_RECONFIG(RECONFIG_NOTHING_CHANGED);

    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Bfp8_b
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Bfp16_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;    // Bfp16_b
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Fp32
    constexpr auto cb_out1 = tt::CBIndex::c_17;  // Bfp8_b

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    /*
     * For each init test, it will cover different paths for state tracker reconfiguration calls
     */

    // --- UNARY init test ---
    WAYPOINT("BNRY");
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_NOTHING_CHANGED));

    binary_op_init_common(cb_in1, cb_in2, cb_out1);  // All 3 changed
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    binary_op_init_common(cb_in0, cb_in1, cb_out1);  // SRCA + SRCB changed
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    binary_op_init_common(cb_in2, cb_in1, cb_out1);  // Only SRCA changed
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    binary_op_init_common(cb_in2, cb_in0, cb_out1);  // Only SRCB changed
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCB));

    binary_op_init_common(cb_in2, cb_in0, cb_out0);  // Only PACK changed
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_PACK));
}
}  // namespace NAMESPACE
