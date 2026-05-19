// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensix consumer kernel for borrowed-memory DFB tests.
// Pops entries from the DFB ring so the DM producer's finish() can complete.
// Host verifies the data by reading the borrowed L1 buffer directly.
//
// Compile-time args:
//   CTA[0]: num_entries  - total number of entries to consume

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_compile_time_arg_val(0);
    DataflowBuffer dfb(0);

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
}
