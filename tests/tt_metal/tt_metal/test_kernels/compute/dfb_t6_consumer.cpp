// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(1);
    constexpr uint32_t blocked_consumer = get_compile_time_arg_val(2);

    uint32_t logical_dfb_id = get_arg_val<uint32_t>(1);

    experimental::DataflowBuffer dfb(logical_dfb_id);

    // Each consumer pops exactly num_entries_per_consumer entries from its own TC(s).
    // No modulo-skip is needed: the DFB hardware delivers only this consumer's entries
    // to its TC, so every wait_front/pop_front here is for a tile this consumer owns.
    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        dfb.wait_front(1);
        DEVICE_PRINT_UNPACK("unpack consumer tile id {}\n", tile_id);
        dfb.pop_front(1);
        DEVICE_PRINT_PACK("pack consumer tile id {}\n", tile_id);
    }
    DEVICE_PRINT("CBWW\n");
    dfb.finish();
    DEVICE_PRINT("CBWD\n");
}
