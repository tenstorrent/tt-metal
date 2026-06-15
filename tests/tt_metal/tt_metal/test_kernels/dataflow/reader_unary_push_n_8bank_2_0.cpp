// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t ublock_size_tiles = get_arg(args::ublock_size_tiles);
    bool reader_only = get_arg(args::reader_only);

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    uint32_t tile_bytes = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(ta::src_tensor);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        if (reader_only == false) {
            dfb.reserve_back(ublock_size_tiles);
        }
#ifdef ARCH_QUASAR
        uint32_t l1_write_addr = dfb.get_write_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR;
#else
        uint32_t l1_write_addr = dfb.get_write_ptr();
#endif

        for (uint32_t tile = 0; tile < ublock_size_tiles; tile++) {
            uint64_t src_noc_addr = get_noc_addr(i + tile, tensor_accessor);
            noc_async_read(src_noc_addr, l1_write_addr + tile * tile_bytes, tile_bytes);
        }
        noc.async_read_barrier();
        if (reader_only == false) {
            dfb.push_back(ublock_size_tiles);
        }
    }
}
