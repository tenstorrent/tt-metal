// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, fast sort-core form (RTF_SPLIT), core B's second data-movement RISC (NOC 1): the 16 pre-transposed
// index tiles into cb_index, while the reader RISC fetches the logits.  Runtime arg 0: index template address.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr auto index_args = TensorAccessorArgs<0, 0>();
    const auto index_template = TensorAccessor(index_args, index_addr);
    Noc noc;
    DataflowBuffer index(cb_index);
    FUSED_ZONE("fz_rt_w_index");
    const uint32_t tile_bytes = index.get_entry_size();
    index.reserve_back(Wt);
    for (uint32_t w = 0; w < Wt; ++w) {
        noc.async_read(index_template, index, tile_bytes, {.page_id = w}, {.offset_bytes = w * tile_bytes});
    }
    noc.async_read_barrier();
    index.push_back(Wt);
}
