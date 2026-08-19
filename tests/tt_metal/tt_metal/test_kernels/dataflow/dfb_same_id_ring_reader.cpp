// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    DataflowBuffer initial(dfb::initial);
    DataflowBuffer rhs(dfb::rhs);
    const uint32_t tile_bytes = initial.get_entry_size();

    initial.reserve_back(1);
    noc.async_read(dram_src, initial, tile_bytes, {.bank_id = bank_id, .addr = src_addr}, {});
    noc.async_read_barrier();
    initial.push_back(1);

    src_addr += tile_bytes;
    rhs.reserve_back(1);
    noc.async_read(dram_src, rhs, tile_bytes, {.bank_id = bank_id, .addr = src_addr}, {});
    noc.async_read_barrier();
    rhs.push_back(1);
}
