// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

void kernel_main() {
    const uint32_t out_cb = get_compile_time_arg_val(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_bank_id_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

#ifdef ARCH_QUASAR
    DataflowBuffer cb_out(out_cb);
    uint32_t ublock_size_bytes = cb_out.get_entry_size();
#else
    CircularBuffer cb_out(out_cb);
    uint32_t ublock_size_bytes = cb_out.get_tile_size();
#endif
    // single-tile ublocks
    uint32_t ublock_size_tiles = 1;
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb_out.wait_front(ublock_size_tiles);

        noc.async_write(cb_out, dram_dst, ublock_size_bytes, {}, {.bank_id = dst_dram_bank_id_addr, .addr = dst_addr});

        noc.async_write_barrier();

        cb_out.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
