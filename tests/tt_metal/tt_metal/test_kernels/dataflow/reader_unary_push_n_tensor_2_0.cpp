// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t ublock_size_tiles = get_arg(args::ublock_size_tiles);
    bool reader_only = get_arg(args::reader_only);

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const auto src = TensorAccessor(ta::src_tensor);
    uint32_t ublock_size_bytes = dfb.get_entry_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        if (reader_only == false) {
            dfb.reserve_back(ublock_size_tiles);
        }
        noc.async_read(src, dfb, ublock_size_bytes, {.page_id = i}, {});
        noc.async_read_barrier();
        if (reader_only == false) {
            dfb.push_back(ublock_size_tiles);
        }
    }
}
