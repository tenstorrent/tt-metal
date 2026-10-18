// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/**
 * Minimal kernel used by test_mesh_coords.cpp.
 * The "row" and "col" compile-time args must be the global system-mesh row and column of the device
 * this kernel is running on.  The DEVICE_PRINT line lets the test verify that DEVICE_PRINT output is
 * restricted to the expected mesh coordinate when TT_METAL_DPRINT_MESH_COORDS is configured.
 */
void kernel_main() {
    constexpr uint32_t row = get_arg(args::row);
    constexpr uint32_t col = get_arg(args::col);
    DEVICE_PRINT("mesh_coord=({},{})\n", row, col);
}
