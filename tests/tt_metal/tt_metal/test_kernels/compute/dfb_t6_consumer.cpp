// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // CTA layout mirrors dfb_consumer.cpp: [dst_addr, num_entries_per_consumer, blocked_consumer, implicit_sync, ...]
    // dst_addr (CTA[0]) and TensorAccessorArgs are unused by the Tensix consumer.
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(1);
    constexpr uint32_t blocked_consumer = get_compile_time_arg_val(2);

    // RTA layout mirrors dfb_consumer.cpp: [consumer_mask, logical_dfb_id, chunk_offset]
    // chunk_offset (RTA[2]) is unused since Tensix pops from DFB without writing to DRAM.
    uint32_t logical_dfb_id = get_arg_val<uint32_t>(1);

    experimental::DataflowBuffer dfb(logical_dfb_id);

    // DPRINT << "t6 consumer trisc_id: " << trisc_id << " consumer_idx: " << consumer_idx
    //        << " num_entries_per_consumer: " << num_entries_per_consumer << ENDL();
    // DEVICE_PRINT("t6 consumer trisc_id: {} consumer_idx: {} num_entries_per_consumer: {}\n", trisc_id, consumer_idx,
    // num_entries_per_consumer);

    // Each consumer pops exactly num_entries_per_consumer entries from its own TC(s).
    // No modulo-skip is needed: the DFB hardware delivers only this consumer's entries
    // to its TC, so every wait_front/pop_front here is for a tile this consumer owns.
    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        dfb.wait_front(1);
        UNPACK(DPRINT << "unpack consumer tile id " << tile_id << ENDL());
        DEVICE_PRINT_UNPACK("unpack consumer tile id {}\n", tile_id);
        dfb.pop_front(1);
        PACK(DPRINT << "pack consumer tile id " << tile_id << ENDL());
        DEVICE_PRINT_PACK("pack consumer tile id {}\n", tile_id);
    }
    DPRINT << "CBWW" << ENDL();
    dfb.finish();
    DPRINT << "CBWD" << ENDL();
    DEVICE_PRINT("CBWD\n");
}
