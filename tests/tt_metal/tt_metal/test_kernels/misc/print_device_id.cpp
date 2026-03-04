// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

/**
 * Minimal kernel used by test_dprint_mesh_coords.cpp.
 * Compile-time arg 0 must be the host-side chip ID of the device this kernel is running on.
 * The DPRINT line is used by the test to verify that DPRINT output is restricted to the
 * expected device when TT_METAL_DPRINT_MESH_COORDS is configured.
 */
void kernel_main() {
    constexpr uint32_t device_id = get_compile_time_arg_val(0);
    DPRINT << "device_id=" << device_id << ENDL();
}
