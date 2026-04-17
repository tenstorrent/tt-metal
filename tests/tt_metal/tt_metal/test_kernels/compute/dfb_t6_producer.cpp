// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // CTA layout mirrors dfb_producer.cpp: [src_addr, num_entries_per_producer, implicit_sync, blocked_consumer, ...]
    // src_addr (CTA[0]), implicit_sync (CTA[2]), blocked_consumer (CTA[3]), and TensorAccessorArgs are
    // unused: the Tensix producer doesn't do NOC reads. The host pre-fills DFB L1 and the kernel
    // only posts credits so the DM consumer knows data is available.
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(1);

    experimental::DataflowBuffer dfb(0);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        DEVICE_PRINT("producer tile id {}\n", tile_id);
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    DEVICE_PRINT("PFW\n");
    dfb.finish();
    DEVICE_PRINT("PFD\n");
}
