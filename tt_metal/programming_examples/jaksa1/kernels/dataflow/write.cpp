// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t work_per_core = get_arg_val<uint32_t>(1);
    uint32_t work_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_offset = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_offset, out_addr, tile_size_bytes);

    for (uint32_t i = 0; i < work_per_core; i++) {
        cb_wait_front(cb_out, 1);
        DPRINT << "WRITE - TILE " << work_offset + i << " START" << ENDL();
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(work_offset + i, out, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);

        DPRINT << "WRITE - TILE " << work_offset + i << " END" << ENDL();
    }
}
