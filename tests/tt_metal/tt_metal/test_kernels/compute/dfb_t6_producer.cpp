// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(1);

    experimental::DataflowBuffer dfb(0);

    DPRINT << "num_entries_per_producer: " << num_entries_per_producer << ENDL();
    DEVICE_PRINT("num_entries_per_producer: {}\n", num_entries_per_producer);

    // for (uint32_t tensix_id = 0; tensix_id < 4; tensix_id++) {
    //     for (uint32_t tc_id = 0; tc_id < 16; tc_id++) {
    //         DPRINT << "tensix_id: " << tensix_id << " tc_id: " << tc_id
    //                << " capacity: " << static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id)) << ENDL();
    //         DEVICE_PRINT("tensix_id: {} tc_id: {} capacity: {}\n", tensix_id, tc_id,
    //         static_cast<uint32_t>(llk_intf_get_capacity(tensix_id, tc_id)));
    //     }
    // }

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        // DPRINT << "rbw" << ENDL();
        // DEVICE_PRINT("rbw\n");
        dfb.reserve_back(1);
        // DPRINT << "rbd" << ENDL();
        // DEVICE_PRINT("rbd\n");
        // DPRINT << "rdi" << ENDL();
        // DEVICE_PRINT("rdi\n");
        // generate
        DPRINT << "producer tile id " << tile_id << ENDL();
        dfb.push_back(1);
        // DPRINT << "pbd" << ENDL();
        // DEVICE_PRINT("pbd\n");
    }
    DPRINT << "PFW" << ENDL();
    DEVICE_PRINT("PFW\n");
    dfb.finish();
    DPRINT << "PFD" << ENDL();
    DEVICE_PRINT("PFD\n");
}
