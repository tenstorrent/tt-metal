// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/**
 * Minimal kernel used by test_mesh_coords.cpp.
 * Compile-time args 0 and 1 must be the global system-mesh row and column of the device
 * this kernel is running on.  The DEVICE_PRINT line lets the test verify that DEVICE_PRINT output is
 * restricted to the expected mesh coordinate when TT_METAL_DPRINT_MESH_COORDS is configured.
 */
void kernel_main() {
    constexpr uint32_t row = get_compile_time_arg_val(0);
    constexpr uint32_t col = get_compile_time_arg_val(1);
    DEVICE_PRINT("mesh_coord=({},{})\n", row, col);
}
