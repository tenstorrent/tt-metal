// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/kernel_thread_globals.h"
#include <cstdint>

void kernel_main() {
    const uint32_t logical_x = get_absolute_logical_x();
    const uint32_t logical_y = get_absolute_logical_y();
    const uint32_t dm_core_id = get_my_thread_id();

    DPRINT << "DM hello world from logical core (" << logical_x << "," << logical_y << "), dm_core_id=" << dm_core_id
           << "." << ENDL();
}
