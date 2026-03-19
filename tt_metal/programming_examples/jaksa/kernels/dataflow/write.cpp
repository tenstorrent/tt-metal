// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    uint32_t dbg_tile = 42;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_offset = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_offset, c_addr, tile_size_bytes);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            DPRINT << "WRITE - TILE " << mt * Nt + nt << " START" << ENDL();

            cb_wait_front(cb_out, 1);
            uint32_t l1_addr = get_read_ptr(cb_out);
            noc_async_write_tile(mt * Nt + nt, out, l1_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);

            DPRINT << "WRITE - TILE " << mt * Nt + nt << " END" << ENDL();
        }
    }
}
