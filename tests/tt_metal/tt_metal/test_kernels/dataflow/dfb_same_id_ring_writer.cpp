// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t num_wraps = get_arg_val<uint32_t>(2);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;
    DataflowBuffer outbound(dfb::outbound);
    const uint32_t tile_bytes = outbound.get_entry_size();

    for (uint32_t wrap = 0; wrap < num_wraps; ++wrap) {
        outbound.wait_front(1);
        noc.async_write(outbound, dram_dst, tile_bytes, {}, {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        outbound.pop_front(1);
        dst_addr += tile_bytes;
    }
}
